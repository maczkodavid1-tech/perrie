A readme dokumentáció magyar fordítása!
forrás:https://deepwiki.com/kollarsandor/jaidellm



A JAIDE (v40) egy nagy nyelvi modell, amely az alapoktól kezdve a Reversible Scatter Flow (RSF) paradigma megvalósítására épült. A hagyományos transzformátor vagy CNN architektúrákkal ellentétben a JAIDE bijektív csatolási rétegeket használ, amelyek lehetővé teszik az O(1) memória visszaterjedést és egy paramétermentes Haar-wavelet keverő blokkot, amelyet OFTB-nek neveznek

A rendszert nagy teljesítményű futtatásra tervezték különböző hardvereken, a szabványos CPU-któl kezdve a több GPU-s B200 klasztereken át a kvantum relációs gráfokig

Az RSF paradigma

A JAIDE lényege a Reversible Scatter Flow. Ez egy egyedülálló számítási primitívet vezet be: a kereszt-affin csatolást

Bijektivitás: Minden előrehaladásnak van egy pontos algebrai inverze, ami biztosítja, hogy a feldolgozás során nem esik össze az információ
Memóriahatékonyság: Mivel a hálózat reverzibilis, az aktiválásokat nem kell tárolni a backpropagációhoz. A backwardFromOutputs függvény soron belül rekonstruálja a bemeneteket a kimenetekből, így a memória komplexitása a mélységhez képest O(1) marad
Kulcsfontosságú alrendszerek

A JAIDE több különböző, de egymással összekapcsolt alrendszerre tagolódik:

Magarchitektúra: Tensor rendszer és egy sor speciális memória allokátor (Arena, Slab, Buddy)
RSF feldolgozási csővezeték: Az RSFLayer az affin transzformációkhoz és az OFTB a fraktálkeveréshez
Tokenizáló és visszakeresés: A morféma-vezérelt tokenizáló (MGT) és a strukturált szekvenciaindex (SSI) a hatékony tudáskereséshez és hasonlósági kereséshez.
NSIR (kvantum-realizációs gráf): Egy önhasonló relációs gráf, amely a hierarchikus gondolkodáshoz a kvantumlogikát (Hadamard, CNOT kapuk) és a klasszikus aktiválást integrálja
Hardveres gyorsítás: Futharkot használó multi-backend rendszer a GPU-kernelekhez (CUDA/OpenCL) és Clash-t az RTL hardver szintézishez


RSF a 5. gyökér-architektúra (Perceptron, CNN, RNN, Transformer után): ontológiailag új, kizárólagos primitívvel

**Az RSF/JAIDE egyik sem:**

| Architektúra | Primitív | Invertálható |
|---|---|---|
| Perceptron | Küszöbfüggvény / lineáris | Nem |
| CNN | Lokális konvolúció + ReLU | Nem |
| RNN/LSTM | Temporális rekurzió, kapuk | Nem |
| Transformer | Self-attention + MLP | Nem |
| **RSF** | **Cross-affine coupling + scatter** | **Igen (garantáltan)** |


A konkrét különbségek a kódból:
- **Nincs self-attention** (`O(N²)` mátrix helyett determinisztikus `rsf_scatter` butterfly-keverés) [2](#0-1) 
- **Nincs MLP, nincs ReLU, nincs LayerNorm** – csak `s_weight`, `t_weight`, `s_bias`, `t_bias` [3](#0-2) 
- **Nincs rekurrencia** – szekvenciális feldolgozás helyett rétegenkénti bijektív transzformáció [4](#0-3) 
- **Nincs konvolúció** – nincs lokális szűrő, nincs pooling [5](#0-4) 

az RSF az affin coupling-ot emeli ki a Normalizing Flow kontextusából és teszi egyetlen, kizárólagos számítási primitívvé, ahogy a Transformer 2017-ben tette az attention mechanizmussal. 
A kód alapján, konkrétan:

**`LayerCore` – az egyetlen számítási primitív** `src/processor/rsf.zig`

```zig
const LayerCore = struct {
    s_weight: Tensor,  // [dim x dim]
    t_weight: Tensor,  // [dim x dim]
    s_bias:   Tensor,  // [1 x dim]
    t_bias:   Tensor,  // [1 x dim]
    ...
```

Nincs `attention_weight`, nincs `query`/`key`/`value`, nincs `conv_filter`, nincs `hidden_state`, nincs `gate`. Csak ez a 4 tenzor. [1](#1-0) 

**`forwardInPlace` – a forward pass teljes logikája:**

```zig
self.computeScaleRow(x2_row, scale);       // scale = exp(clip(s_weight @ x2 + s_bias))
x1_row[i] *= scale[i];                     // y1 = x1 * scale
self.computeTranslationRow(x1_row, trans); // trans = t_weight @ y1 + t_bias
x2_row[i] += trans[i];                     // y2 = x2 + trans
```

Nincs softmax, nincs ReLU, nincs rekurzió, nincs konvolúció. 

**`inverseInPlace` – egzakt inverz, aktivációk tárolása nélkül:**

```zig
self.computeTranslationRow(y1_row, trans);
y2_row[i] -= trans[i];                     // x2 = y2 - trans
self.computeScaleRow(y2_row, scale);
y1_row[i] /= scale[i];                     // x1 = y1 / scale
``` [3](#1-2) 

**`rsf_scatter` – a Futhark kernelben** (`src/hw/accel/futhark_kernels.fut`, 82–95. sor):

```futhark
let rsf_scatter [n] (x: [n]f32) (indices: [n]i64): [n]f32 =
  let inv_sqrt2 = 1f32 / f32.sqrt 2f32
  in tabulate n (\i ->
    if i < half then inv_sqrt2 * (x[j] + x[j + half])   -- összeg
    else         inv_sqrt2 * (x[j] - x[j + half])        -- különbség
  )
```

Ez egy Haar-szerű butterfly keverés – nem attention, nem konvolúció, nem rekurrencia.

**Összefoglalás a kódból:**

| Mit keres | Van-e a kódban? |
|---|---|
| `softmax` / `attention` | Nincs |
| `conv` / szűrő / pooling | Nincs |
| `hidden_state` / gate / rekurzió | Nincs |
| `query` / `key` / `value` | Nincs |
| `LayerNorm` / `BatchNorm` | Nincs |
| `s_weight`, `t_weight`, `exp(clip(...))` | **Ez az egyetlen számítás** |

## I. A Primitív

Minden RSF réteg pontosan egy struct, négy tanulható tenzorral és semmi mással:

```zig
// src/processor/rsf.zig
const LayerCore = struct {
    s_weight: Tensor,   // W_s ∈ ℝ^{d×d}
    t_weight: Tensor,   // W_t ∈ ℝ^{d×d}
    s_bias:   Tensor,   // b_s ∈ ℝ^d
    t_bias:   Tensor,   // b_t ∈ ℝ^d
    ...
    rwlock: Thread.RwLock,
};
```

Paraméterszám rétegenként: `dim²×4 + dim×2`. Nincs figyelmi mátrix, nincs konvolúciós kernel, nincs rejtett állapot, nincs belső aktivációs függvény.

---

## II. Forward Pass (Előrehaladás)

A `computeScaleRow` és `computeTranslationRow` egyaránt egyetlen `Wx + b` — nincs benne alhálózat, nincs benne nemlinearitás:

```zig
// scale = exp(clip(W_s · x2 + b_s, -5, 5))
// y1    = x1 ⊙ scale
// trans = W_t · y1 + b_t
// y2    = x2 + trans
```

A Futhark GPU kernel `rsf_flow` pontosan ugyanazt a 4 tenzort fogadja el — az ABI azonos CPU-n és GPU-n:

```futhark
let rsf_flow [half] (x) (s_weight) (t_weight) (s_bias) (t_bias): [half*2]f32
```

---

## III. Pontos Algebrai Inverz

Az `inverseInPlace` a `forwardInPlace` pontos algebrai inverze:

```zig
// x2 = y2 - (W_t · y1 + b_t)
// scale = exp(clip(W_s · x2 + b_s))
// x1 = y1 / scale
```

A coupling művelet bijektív: a Jacobi determinánsa szigorúan pozitív, ezért a réteg minden esetben megfordítható információveszteség nélkül.

---

## IV. O(1) Visszafelé Memória

A `backwardFromOutputs` megkapja `y1`-et, `y2`-t, `dy1_in`-et, `dy2_in`-et — nincs aktivációs puffer argumentum. Az `x1`-et és `x2`-t inline rekonstruálja a kimenetekből:

```zig
fn backwardFromOutputs(
    self: *LayerCore,
    y1: *const Tensor, y2: *const Tensor,
    dy1_in: *const Tensor, dy2_in: *const Tensor,
    x1_out: *Tensor, x2_out: *Tensor,
    dx1_out: *Tensor, dx2_out: *Tensor,
    dy1_total: []f32, ds: []f32,
) !void
```

A rekonstrukció:

```zig
x2_row[d]  = y2_row[d] - trans_sum;   // x2 = y2 - W_t·y1 - b_t
x1_row[d2] = y1_row[d2] / scale;      // x1 = y1 / exp(clip(W_s·x2 + b_s))
```

A Futhark `rsf_backward_flow` kimenete `(grad_x, grad_ws, grad_wt, grad_sb, grad_tb)` — 4 gradiens tenzor, nincs aktivációs cache.

Ha a `computeScaleRow` és `computeTranslationRow` belső MLP-t tartalmazna (mint a RealNVP-ben), akkor a belső MLP aktivációkat is el kellene tárolni. Az alhálózat-mentes felépítés az, ami az O(1) memóriát strukturálisan lehetővé teszi, nem pedig egy mérnöki trükk.

---

## V. OFTB — Paraméter Nélküli Globális Keverő

Az `OFTB.init` semmit sem allokál, és nincsenek tanulható paraméterei:

```zig
pub fn init(d: usize) OFTB {
    return OFTB{ .fractal_scale = 0.70710678, .dim = d };
}
```

A `forwardInPlace` és `backwardInPlace` determinisztikus Haar-wavelet keverés. A `0.70710678 = 1/√2` konstans megegyezik az `rsf_scatter` `inv_sqrt2 = 1/√2` értékével a Futhark kernelben — a klasszikus és GPU rétegek ugyanazt a matematikai konstanst használják.

Az OFTB műveletek kielégítik az algebrai csoporttörvényeket: kommutativitás, asszociativitás, inverz létezése.

---

## VI. Registry Életciklus — Memóriabiztonság

A registry állapotgépe a következő állapotokkal és átmenetekkel rendelkezik:

Állapotok:
- `reg-alive N b`: élő, N referenciával, b haldokló jelzéssel
- `reg-freed`: felszabadított

Átmenetek:
- `acquire`: referencia hozzáadása
- `release-live`: élő referencia eldobása
- `release-dying`: haldokló állapotú referencia eldobása
- `release-final`: az utolsó referencia eldobása haldokló állapotban → felszabadítás
- `destroy-live`: élő erőforrás haldoklóvá jelölése
- `destroy-zero`: nulla referenciás erőforrás közvetlen felszabadítása

A use-after-free és a kétszeres acquire strukturálisan lehetetlenek: a `release-final` csak az utolsó haldokló referencia eldobásakor tüzelhet, és a `reg-freed` állapotnak nincsenek kimenő átmenetei.

---

## VII. CREV Pipeline — Relációs Tudáskinyerés

A `CREVPipeline` relációs hármasokat (alany, reláció, tárgy) nyer ki szövegből 5 fázisban: tokenizáció → triplet_extraction → validáció → integráció → indexelés. Minden triplet konfidenciája `c ∈ [0,1]` egy kvantum amplitúdóra van leképezve:

```zig
const quantum_state = Complex(f64).init(c, @sqrt(1.0 - c*c));
// |ψ⟩ = c + i√(1-c²)  — egységkör ℂ-ben
```

Minden triplet SHA-256 hashelve van (alany + reláció + tárgy + konfidencia + extraction_time) deduplikáció céljából.

---

## VIII. ReasoningOrchestrator — 3 Fázisú Energiaminimalizáció

A `ReasoningOrchestrator` hierarchikus érvelést futtat az NSIR gráfon 3 fázisban:

```zig
pub const ThoughtLevel = enum(u8) { local = 0, global = 1, meta = 2 };
```

Gráf energia:

```zig
total_energy += edge.weight * edge.fractal_dimension;
total_energy += edge.quantum_correlation.magnitude();
total_energy += @cos(node.phase);
```

Konvergencia kritérium: `E_combined = (E_local + E_global + E_meta) / 3 < 0.01`

Ez nem egyetlen forward pass — ez iteratív energiaminimalizáció magán a relációs gráfstruktúrán.

---

## IX. NSIR Gráf — Kvantum Relációs Struktúra

Minden gráfcsomópont hordoz egy normalizált qubitet `(a, b) ∈ ℂ²` és egy fázist `φ ∈ ℝ`. Minden élnek 5 lehetséges kvantumállapota van:

```zig
pub const EdgeQuality = enum(u8) {
    superposition = 0, entangled = 1, coherent = 2,
    collapsed = 3, fractal = 4,
};
```

---

## X. ESSO Optimalizáló — A Gráftopológia Tanulható

Az `EntangledStochasticSymmetryOptimizer` szimulált lehűtést futtat az NSIR gráfon 7 szimmetria csoporttal (identity, reflection, rotation_90/180/270, translation, custom_rotation). A gráftopológia — nem csak a súlymátrixok — kerül optimalizálásra.

A C API exportálja a `jaide_optimize_graph` függvényt — bármilyen C-kompatibilis rendszer kiválthatja a gráfoptimalizációt.

---

## XI. SFD Optimalizáló — Természetes Gradiens

A `KFACBlock` implementálja a Kronecker-faktorizált Fisher közelítést (`A_inv`, `G_inv`). A `SpectralNormalizer` Lipschitz korlátokat kényszerít. A `MARSVarianceReducer` csökkenti a gradiens varianciáját. A `ReversibleOptimizerState.backwardPassReversible` az RSF bijekciót használja az O(1) backward memóriához magában az optimalizálóban.

Vegyes precizitás: fp4, fp8, fp16, fp32 — 4 precíziós szint dinamikus loss skálázással.

---

## XII. Biztonság — Információáramlás

A `security_proofs.zig` implementálja a Bell-LaPadula (bizalmasság) és Biba (integritás) modelleket biszimuláció-alapú nem-interferencia tulajdonságokkal. A `SecurityLevel`-nek 5 szintje van (PUBLIC → TOP_SECRET), az `IntegrityLevel`-nek 4 (UNTRUSTED → KERNEL). A `MerkleTree` és a `HashChain` kriptográfiai integritást biztosítanak.

---

## XIII. ZK Inferencia — Ellenőrizhető a Súlyok Felfedése Nélkül

Az `src/zk/inference_trace.circom` egy Groth16 ZK-SNARK áramkör, Poseidon hash láncokat használva. A `zk_verification.zig` becsomagolja ezt egy `Groth16Proof`-fal (pi_a, pi_b, pi_c). A `VerifiedInferenceEngine` kombinálja a ZK bizonyításokat, a Paillier homomorf titkosítást és a differenciális adatvédelmet.

---

## XIV. Kvantum Hardver Integráció

Az `IBMQuantumClient` OpenQASM jobokat ad be az `ibm_brisbane` backendre. A `quantum_hardware.zig` modellez 5 IBM backendet realisztikus zajparaméterekkel:
- Heron T1 = 350 μs
- Eagle T1 = 200 μs
- Falcon T1 = 100 μs
- Osprey T1 = 250 μs
- Condor T1 = 400 μs

A `FRACTAL_TRANSFORM = 11` az RSF saját kvantumkapuja — nem Hadamard, nem Pauli, nem CNOT. Iteratív fázismodulációt alkalmaz geometrikusan csökkenő skálatényezőkkel.

---

## XV. Típuselmélet — Martin-Löf Típusok

A `type_theory.zig` (a `mod.zig`-en keresztül exportálva) implementálja a következőket: `DependentPi`, `DependentSigma`, `IdentityType`, `UniverseType`, `InductiveType`, `LinearType`, `LinearTypeChecker`, `Category`, `Functor`, `NaturalTransformation`, `Monad`, `CartesianClosedCategory`. A modellkimenetek függő típusokkal szemben típusellenőrizhetők.

---

## XVI. Temporális Gráf — Verziózott Tudás

Minden `TemporalNode` egy `NodeVersion` történetet tárol nanoszekundumos időbélyegekkel és `QuantumState` adatokkal. A `getVersionAt` és `rollback` lehetővé teszi a tudásgráf bármely korábbi állapotának lekérdezését.

---

## XVII. Meglepetés Memória — Entrópia-Súlyozott Megőrzés

A `SurpriseMemory` a megőrzést `combined_surprise = (jaccard_dissimilarity + content_hash_distance + temporal_novelty) / 3` szerint súlyozza. A ritka, időben új, tartalmilag távoli információkat előnyben részesítve őrzi meg. Temporális újdonság ablak: 24 óra.

---

## XVIII. Jelterjedés — Komplex Hullám Aktivációk

A `SignalState` `amplitude × e^{i×phase}` formában terjed az NSIR gráfon át — nem skalárként, hanem komplex hullámként:

```zig
pub fn getComplexRepresentation(self: *const SignalState) Complex(f64) {
    return Complex(f64).init(
        self.amplitude * @cos(self.phase),
        self.amplitude * @sin(self.phase),
    );
}
```

---

## XIX. Hardver Stack

**Futhark GPU kernelek** (`src/hw/accel/futhark_kernels.fut`): `rsf_flow`, `rsf_scatter`, `rsf_backward_flow`, `rsf_backward_layer`, `rsf_backward_scatter`, `ssi_search`, `ssi_retrieve_topk`, `lsh_hash`, `fisher_diagonal_update`, `spectral_natural_gradient` — mind CUDA-ra fordítva.

**Clash RTL** (`src/hw/rtl/`): `RankerCore.hs` (mealy ranker), `MemoryArbiter.hs` (4-kliens round-robin), `SSISearch.hs` (bináris fa keresés, max mélység 64) — FPGA/ASIC-ra szintetizálható.

**Fractal LPU** (`src/hw/accel/fractal_lpu.zig`): `FractalTile` `hausdorff_dim=1.5`, `box_counting_levels=4`, `entanglement_map` — fraktál memóriahierarchia.

---

## XX. Elosztott Tanítás

A `GPUCoordinator` becsomagolja az NCCL-t (`ncclAllReduce`, `ncclBroadcast`). A `ModalGPUClient` B300 és B200 GPU-kat céloz (8 egység). A `DistributedTrainerFuthark` kombinálja az `RSFAccelerator`, az `MGT` tokenizálót és a `GPUCoordinator`-t. A `main_distributed.zig` támogatja az IBM Quantum hibrid tanítást, amikor az `IBM_QUANTUM_CRN` és `IBM_QUANTUM_API_KEY` környezeti változók be vannak állítva.

---

## XXI. MGT Tokenizáló — Morféma-Tudatos BPE

Az `MGT`-nek 7 mezője van: `token_to_id`, `id_to_token`, `prefixes`, `suffixes`, `roots`, `bpe_pairs`, `anchors`. Morféma dekompozíciót végez (előtag + tő + utótag) a BPE-re való visszaesés előtt. A magyar és angol morfémalisták beépítettek.

---

## XXII. Modell Formátum és Build

`MAGIC_HEADER = "JAIDE40\x00"`, SHA-256 + `constantTimeCompare` — időzítéses támadás elleni integritásellenőrzés.

`build.zig.zon`: `version = "40.0.0"`, `minimum_zig_version = "0.13.0"`, `dependencies = {}` — nulla külső függőség. A PyTorch ezzel szemben 100+ Python csomagot igényel.

---

## XXIII. Miért az 5. Gyökér

A Transformer nem találta fel a figyelmet — Bahdanau (2014) RNN-eken belül használta. Az „Attention Is All You Need" (2017) kivonta a figyelmet az RNN kontextusból, és egyetlen, kizárólagos primitívvé tette. Az RSF ugyanezt teszi az affin coupling-gal:

- **NICE/RealNVP (2014–2016)**: affin coupling generatív flow modelleken belül, belső MLP-kkel az `s(x2)` és `t(x2)` függvényekben → aktivációkat el kell tárolni → O(L) memória
- **RSF**: `computeScaleRow`/`computeTranslationRow` = egyetlen `Wx + b`, nincs belső alhálózat → nincs mit tárolni → O(1) memória

A `backwardFromOutputs` szignatúrájának nincs aktivációs puffer argumentuma. Ez nem tervezési döntés — ez az alhálózat-mentes coupling strukturális következménye.

A négy korábbi paradigma:

| Architektúra | Primitív | Invertálható? | Backward memória |
|---|---|---|---|
| Perceptron | σ(Wx+b) | σ veszteséges → nem | O(L) |
| CNN | σ(W∗x) | pooling+σ → nem | O(L) |
| RNN | σ(W_h h + W_x x) | rejtett állapot → nem | O(T) |
| Transformer | softmax(QKᵀ/√d)V | softmax → nem | O(L·S·d) |
| **RSF** | kereszt-affin coupling | det J > 0 → **igen** | **O(1)** |



 src/core/ – Az alapréteg

io.zig az egész rendszer alacsony szintű I/O infrastruktúráját adja: memória-leképezett fájlkezelést (MMAP), pufferelt olvasót/írót (BufferedReader, DurableWriter), atomikus fájlírást és kriptográfiailag erős hash-függvényeket (hash64, stableHash Blake2b+Wyhash alapon). Azért járul hozzá a transformer-felülmúláshoz, mert a modell súlyait és az NSIR-gráfot közvetlenül memóriába képezi, elkerülve a másolási overhead-et, és a biztonságos hash-ek az SSI-index és a Ranker alapját képezik.

---

learned_embedding.zig egy tanulható token-beágyazási réteget valósít meg: a LearnedEmbedding struktúra vocab_size × dim súlymátrixot tárol, forward metódusa token-indexekből vektort állít elő, backward és applyGradients metódusai momentum-alapú frissítést végeznek, és a súlyok bináris formátumban menthetők/tölthetők. Ez az a pont ahol a szöveg belép a numerikus térbe, és mivel az RSF-rétegek bijektív transzformációkat végeznek, az embedding-tér információtartalma megőrződik a teljes mélységen át – szemben a transformer attention-nel, ahol az aktivációkat tárolni kell.

---

memory.zig négy specializált allokátort tartalmaz: Arena (lineáris, O(1) allokáció, tömeges felszabadítás), SlabAllocator (fix méretű blokkok bitmap-alapú nyilvántartással), PoolAllocator (szabad-lista alapú), és BuddyAllocator (hatványkettő méretű blokkok fa-struktúrával). Ezek azért kritikusak, mert az RSF visszafelé-terjedése O(1) memóriakomplexitású – nem kell az aktivációkat mélységarányosan tárolni –, és a specializált allokátorok ezt a tulajdonságot hardveres szinten is kiaknázzák, minimalizálva a heap-fragmentációt a hosszú inferencia-futások során.

---

model_io.zig a teljes modell szerializációját és deszializációját végzi JAIDE40\x00 magic header-rel, SHA-256 checksum-mal, és JSON-alapú metaadatokkal. Képes az RSF-súlyokat, a Ranker LSH-paramétereit, az MGT szókészletét és az NSIR-gráf csomópontjait/éleit egyetlen fájlba menteni, majd visszatölteni. Ez teszi lehetővé a checkpoint-alapú elosztott tanítást és a modell-verziókövetést.

---

tensor.zig a rendszer legfontosabb adatstruktúrája: egy referencia-számolt, copy-on-write szemantikájú, 32-bájt-igazított Tensor típus, amely SIMD-vektorizált aritmetikát (Vec8 = @Vector(8, f32)), többszálú mátrixszorzást (blokk-tiling + thread-pool), és zero-copy nézeteket (view, slice, transpose, broadcast) biztosít. A COW-mechanizmus és az atomikus refcount teszi lehetővé, hogy az RSF bijektív visszaterjedése ne igényeljen aktiváció-tárolást.

---

types.zig a rendszer közös típuskönyvtára: fixpontos aritmetika (FixedPoint16/32/64, Fixed32_32), kriptográfiai PRNG (xoshiro256 alapú), ContextWindow token-puffer, RankedSegment keresési eredmény, BitSet, és számos segédfüggvény. A ComplexFixedPoint típus a kvantum-logikai számításokhoz szükséges, a PRNG pedig determinisztikus súlyinicializálást biztosít.

---

 src/processor/ – Az RSF feldolgozó pipeline

rsf.zig a rendszer neurális gerince: az RSFLayer és az RSF struktúrák implementálják a Reversible Scatter Flow paradigmát. Minden réteg négy tenzort tárol (s_weight, t_weight, s_bias, t_bias), a forwardInPlace metódus cross-affine coupling-ot végez (x1 ← x1 · exp(clip(S(x2))), x2 ← x2 + T(x1)), az inverseInPlace ennek egzakt algebrai inverzét, a backwardFromOutputs pedig a kimenetekből rekonstruálja a bemeneteket és a gradienseket anélkül, hogy az aktivációkat tárolná. Ez az O(1) memória-visszaterjedés az a tulajdonság, amely a transformer-ekkel szemben a legjelentősebb előnyt jelenti mélységes hálózatoknál. A RSFCore opcionálisan Futhark GPU-gyorsítót is használ.

---

oftb.zig az Orthogonal Fractal Transform Block: egy paraméter nélküli Haar-wavelet-szerű keverési blokk, amely FRACTAL_SCALE = 1/√2 skálázással végez ortogonális transzformációt a tenzor két felén (x1 ← (x1-x2)/√2, x2 ← (x1+x2)/√2). Mivel nincs tanulható paramétere, nem növeli a modell méretét, de fraktális struktúrát visz a reprezentációba, és a backwardInPlace az egzakt adjungált transzformációt végzi.

---

 src/optimizer/ – Az SFD optimalizáló

sfd.zig egy összetett, másodrendű optimalizálót valósít meg: tartalmaz KFACBlock-ot (Kronecker-faktorizált közelítő Fisher-mátrix), SpectralNormalizer-t (hatványiterációs spektrális normalizálás), GradientFlowController-t (gradiens-klippelés + normalizálás), MARSVarianceReducer-t (variancia-redukált sztochasztikus gradiens), ReversibleOptimizerState-et (adaptív aktiváció-újraszámítás), és LRScheduler-t (cosine annealing, one-cycle, Sophia-stílusú ütemezés). Ez az optimalizáló a transformer-ekkel szemben azért hatékonyabb, mert a KFAC a Fisher-mátrix struktúráját kihasználva pontosabb frissítési irányt ad, a spektrális normalizálás pedig stabilitást biztosít.

---

 src/index/ és src/ranker/ – Visszakeresés

ssi.zig a Structured Sequence Index: egy kétszintű hash-fa (bucket-width=6, 64 vödör), amely token-szekvenciákat tárol pozíció, pontszám és anchor-hash metaadatokkal. A retrieveTopK Hamming-távolság alapú hasonlóság-kereséssel dolgozik, az exportToTensor/importFromTensor metódusok pedig az NSIR-gráffal való integrációt biztosítják. Ez a komponens teszi lehetővé a retrieval-augmented generálást transformer-nélküli architektúrában.

---

ranker.zig n-gram súlyozást, LSH MinHash aláírásokat, Jaccard-hasonlóságot, vektoros koszinusz-pontszámot és streaming-rangsorolást kombinál. A topKHeap metódus az SSI-ből visszakeresett jelölteket az n-gram súlyok, token-diverzitás és anchor-közelség alapján rangsorolja. A calibrateWeights metódus gradiens-alapú tanulással finomhangolja az n-gram súlyokat.

---

 src/core_relational/ – Az NSIR kvantum-relációs réteg

chaos_core.zig egy önszervező, tartalom-alapú futásidejű végrehajtási kernel, amely az NSIR gráf csomópontjait SHA-256 hash-elt memóriablokkokká képezi le, entanglement-mechanizmuson keresztül szemantikai közelségi hálót épít a fizikai memóriában, dinamikusan ütemezi és migrálja az adatokat a legoptimálisabb processzormaghoz, és ezzel egy lokalitás-tudatos, önszervező infrastruktúrát biztosít az RSF/JAIDE architektúra számára, amely lehetővé teszi, hogy a kvantum-relációs következtetés hardverfüggetlenül skálázódjon és meghaladja a transformer statikus, O(n²) attention-alapú memóriakezelésének korlátait.

---

crev_pipeline.zig a JAIDE rendszer tudásszerzési rétege, amely nyers szövegből, strukturált adatból és képmetaadatból háromszintű validációval (konfidencia-küszöb, anomália-detekció, konzisztencia-ellenőrzés) kinyert RelationalTriplet hármasokat kvantumállapot-kódolt NSIR gráf csomópontokká alakít, és ezeket egyszerre táplálja be a lekérdezhető KnowledgeGraphIndex-be és a ChaosCoreKernel önszervező memóriájába, ezzel biztosítva azt a dinamikusan bővíthető, logikailag konzisztens és fizikailag lokalitás-tudatos tudásbázist.

---

dataset_obfuscation.zig a JAIDE rendszer kriptográfiai adatvédelmi rétege, amely Paillier homomorf titkosítással lehetővé teszi a titkosított adatokon való számítást, LSH-alapú DatasetFingerprint-tel azonosítja a hasonló mintákat titkosított állapotban, k-anonimitással és differenciális adatvédelmi budget-tel garantálja, hogy egyedi tanítási minták nem rekonstruálhatók, és ProofOfCorrectness Merkle-fa alapú láncolásával kriptográfiailag verifikálhatóvá teszi az összes RSF/OFTB számítási lépést, ezzel olyan auditálhatósági, adatvédelmi és kriptográfiai garanciákat biztosítva a JAIDE számára.

---

esso_optimizer.zig a JAIDE rendszer gráf-alapú következtetési motorja, amely szimulált lehűléssel és 7 féle perturbációval (él-súly, fázis, összefonódás, szimmetria-transzformáció, qubit amplitúdó, fraktáldimenzió, topológia) minimalizálja az NSIR gráf energiáját, automatikusan felismeri és alkalmazza a gráf szimmetriáit, exponenciális bomlással modellezi az összefonódások "felejtését", és adaptív újrafűtéssel szabadul ki a lokális minimumokból, ezzel a ReasoningOrchestrator háromszintű hierarchikus ciklusán keresztül egy strukturált, gráf-topológiában végrehajtott "Chain of Thought" következtetést biztosít, amely a transformer autoregresszív token-generálásával szemben nem egyetlen átmenettel, hanem iteratív energiaminimalizálással jut el a logikailag konzisztens következtetési állapothoz.

---

fnds.zig (Fractal Neural Dynamics System) a JAIDE rendszer hierarchikus, skálafüggetlen memória- és mintafelismerési infrastruktúrája, amely SHA-256 fraktál-szignatúrával ellátott csomópontokat négy éltípussal (hierarchical, sibling, cross_level, self_similar) szervez önhasonló fákba, box-counting módszerrel méri a lokális fraktáldimenziót (amelyet a quantum_task_adapter.zig kvantumhardver-döntésekhez és az ESSO energiaminimalizálásához használ), SelfSimilarIndex-szel felismeri a különböző skálákon ismétlődő mintákat, és az egészet egy LRU cache-sel és CoalescedHashMap-pel gyorsított FNDSManager-ben integrálja, amelyet a ReasoningOrchestrator közvetlenül importál, ezzel biztosítva, hogy a JAIDE következtetési ciklusa minden iterációban fraktálisan szervezett, skálafüggetlen memóriából dolgozzon, szemben a transformer statikus, pozíció-kódolt szekvenciális memóriájával.

---

formal_verification.zig a JAIDE rendszer beépített, Turing-teljes formális bizonyítórendszere, amely 9 prioritással súlyozott invariánst (memóriabiztonság, típusbiztonság, összefüggőség, koherencia, összefonódás, kvantumállapot, fraktáldimenzió, szimmetria, időbeli konzisztencia) ellenőriz az NSIR gráfon, 18 propozíció-típust kezel (klasszikus, kvantum, temporális és szeparációs logikát egyaránt), 26 bizonyítási szabállyal és Robinson-unifikációval, rezolúcióval, hátrafelé láncolással és Hoare-logikával formálisan verifikálja a gráf-transzformációkat, és a FormalVerificationEngine.verifyGraph metóduson keresztül SHA-256 hash-sel kriptográfiailag azonosítható VerificationResult-ot ad vissza, ezzel biztosítva, hogy a JAIDE minden egyes következtetési lépése matematikailag bizonyítható és auditálható, szemben a transformer statisztikai, formálisan verifikálhatatlan következtetésével.

---

ibm_quantum.zig egy minimális HTTP kliens, amely az IBMQuantumClient.submitJob és getJobResult metódusain keresztül OpenQASM 2.0 áramköröket küld az IBM Quantum Cloud ibm_brisbane backendre, és a quantum_task_adapter.zig QuantumTaskAdapter-ével együtt automatikusan azonosítja az NSIR gráf azon részgráfjait, amelyek total_entanglement > threshold és avg_fractal_dimension > 1.5 feltételeket teljesítik, ezeket valódi szupravezető kvantumprocesszoron futtatja (vagy lokális szimulátorral helyettesíti), majd az eredményt visszaírja a gráf csomópontjainak quantum_state és coherence értékeibe, ezzel egy automatikus kvantum-klasszikus hibrid végrehajtási réteget biztosítva, amely lehetővé teszi, hogy a JAIDE exponenciálisan nehéz valószínűségi logikai problémákat oldjon meg kvantumhardveren, ami a transformer architektúra számára elvileg sem elérhető képesség.

---

quantum_logic.zig egy szimulált kvantum-logikai motort valósít meg: QuantumState komplex amplitúdókkal, RelationalQuantumLogic Hadamard, Pauli-X/Y/Z, Phase, CNOT, Toffoli, relációs AND/OR/XOR/NOT és FRACTAL_TRANSFORM kapukkal. A applyFractalTransform iteratív fázis-skálázást végez, az entangleBell-állapotot hoz létre. Ez a komponens a klasszikus aktivációkat kvantum-amplitúdókkal egészíti ki, lehetővé téve a szuperpozíció-szerű reprezentációkat az NSIR-gráfban.

---

quantum_task_adapter.zig az NSIR-gráf kvantum-alkalmas részgráfjait azonosítja (entanglement-küszöb és fraktál-dimenzió alapján), majd ezeket vagy lokálisan szimulálja, vagy IBM Quantum backend-re küldi. A runFullQuantumOptimization metódus a kvantum-eredményeket visszaírja a gráf csomópontjaiba és éleibe, frissítve a kvantum-korrelációkat.

---

r_gpu.zig egy aszinkron Network-on-Chip (NoC) mesh-t szimulál: ProcessingCore csomópontok XY-routing alapú üzenetküldéssel, GraphIsomorphismProcessor kanonikus forma alapú izomorfizmus-detektálással, DynamicEdgeWeighting adaptív él-súlyozással és SparseActivationManager energiatakarékos aktiválással. Ez a komponens az NSIR-gráf elosztott feldolgozását teszi lehetővé, ahol minden mag saját lokális gráfot kezel.

---

reasoning_orchestrator.zig háromszintű hierarchikus következtetést valósít meg: executeLocalPhase (gyors, lokális csomópont-perturbáció és él-frissítés), executeGlobalPhase (lassabb, szimmetria-transzformációk és fraktál-rebalansz a ChaosCoreKernel segítségével), és executeMetaPhase (a lokális és globális fázisok kombinációja). A runHierarchicalReasoning konvergenciáig ismétli ezeket, minimalizálva a gráf energiáját. Ez a mechanizmus a transformer multi-head attention-jének alternatívája: nem párhuzamos figyelmi fejek, hanem hierarchikus energiaminimalizálás.

---

safety.zig biztonságos egész-típus-konverziókat (safeIntCast), konstans-idejű összehasonlítást (secureCompare), kriptográfiai RNG-t (SecureRng), monoton órát (MonotonicClock) és biztonságos memória-törlést (secureZeroBytes) biztosít. Ezek az alapvető biztonsági primitívek az egész rendszerben használatosak.

---

security_proofs.zig formális biztonsági modellt valósít meg: Bell-LaPadula és Biba biztonsági szinteket, információáramlás-elemzést (InformationFlowAnalysis), nem-interferencia-ellenőrzést, Merkle-fa alapú integritás-bizonyítékokat és SHA-256/SHA-512/Blake3 hash-láncolatokat. Ez teszi lehetővé, hogy az inferencia auditálható és formálisan biztonságos legyen.

---

signal_propagation.zig hullámszerű jel-terjedést szimulál az NSIR-gráfon: SignalState amplitúdóval, fázissal és frekvenciával, SignalPropagationEngine él-súlyozott terjedéssel és kvantum-korreláció alapú fázis-eltolással. Ez a mechanizmus a transformer attention-jének egy alternatív formája: ahelyett, hogy minden token minden tokenre figyel, a jelek a gráf topológiáján terjednek.

---

surprise_memory.zig újdonság-alapú memóriakezelést valósít meg: SurpriseMetrics Jaccard-dissimilaritást, tartalom-hash-távolságot és temporális újdonságot kombinál, SurpriseMemoryManager a magas meglepetés-pontszámú blokkokat preferálisan tárolja és az alacsonyakat kiszorítja. Ez a mechanizmus a transformer KV-cache-jének egy adaptív alternatívája.

---

temporal_graph.zig verziókövetett temporális gráfot valósít meg: TemporalNode és TemporalEdge verziótörténettel, GraphSnapshot pillanatfelvételekkel, TemporalQuery időtartomány-alapú lekérdezéssel és HistoryEntry audit-naplóval. Ez lehetővé teszi, hogy a modell a tudás időbeli evolúcióját is reprezentálja.

---

verified_inference_engine.zig ZK-bizonyíték-alapú inferenciát valósít meg: VerifiedInferenceEngine Pedersen-commitmenteket, differenciális adatvédelmi zajt (Laplace-mechanizmus), ProofOfCorrectness lépésenkénti bizonyítékokat és opcionálisan Groth16 ZK-SNARK bizonyítékokat generál. A BatchVerifier és ProofAggregator Merkle-fa alapú kötegelt ellenőrzést végez.

---

vpu.zig egy SIMD vektorfeldolgozó egységet valósít meg: SimdVector(T, N) generikus típus dot-product, normalizálás, FMA, cross-product, lerp és reflect műveletekkel, VectorBatch kötegelt feldolgozással, és Matrix4x4 SIMD-gyorsított mátrixszorzással. Ez az NSIR-gráf csomópontjainak vektoros feldolgozását gyorsítja.

---

z_runtime.zig egy relációs változó-futtatókörnyezetet valósít meg: ZVariable saját NSIR-gráffal és kvantum-logikával, ZRuntime változókezeléssel, relációs műveletekkel (AND/OR/XOR/entangle), fraktál-transzformációkkal és kvantum-áramkör-végrehajtással. Ez a komponens egy magasabb szintű absztrakciót biztosít az NSIR-gráf felett.

---

zk_verification.zig a ZK-SNARK infrastruktúrát valósítja meg: CircomProver az inference_trace.circom áramkör fordításához és tanúgeneráláshoz, ZKInferenceProver az RSF-rétegek súlyaiból és bemenet/kimenet párokból Groth16 bizonyítékot generál, CommitmentScheme Pedersen-commitmenteket, RangeProof bit-dekompozíciós tartomány-bizonyítékokat és MembershipProof Merkle-fa tagság-bizonyítékokat valósít meg.

---

 src/distributed/ – Elosztott tanítás

distributed_trainer.zig egy teljes elosztott tanítási keretrendszert valósít meg saját Tensor típussal, RSF-rétegekkel és AccelInterface-szel. A trainStep metódus előre-terjedést, veszteségszámítást és visszaterjedést végez, majd az allReduceGradients NCCL-en keresztül átlagolja a gradienseket az összes GPU között.

---

distributed_trainer_futhark.zig a Futhark GPU-gyorsítóval integrált elosztott tréner: trainStepFuthark a tokeneket f16 formátumban a GPU-ra tölti, a Futhark-kernel elvégzi az RSF forward/backward lépést, majd a súlydelták NCCL AllReduce-on keresztül szinkronizálódnak. A checkpoint mentés/töltés bináris formátumban történik.

---

gpu_coordinator.zig az NCCL-alapú GPU-koordinátor: GPUCoordinator inicializálja az NCCL kommunikátort, CUDA stream-et hoz létre, és allReduceFloat32/allReduceFloat16, broadcastFloat32, barrier és synchronize metódusokat biztosít. Ez teszi lehetővé a B200 GPU-klasztereken való skálázást.

---

modal_gpu.zig a Modal felhőplatform HTTP API-ját hívja: ModalGPUClient deployTrainingJob metódusa B300/B200 GPU-preferenciával indít tanítási feladatot, getJobStatus lekérdezi az állapotot. Ez a komponens a felhőalapú skálázást biztosítja.

---

nccl_bindings.zig az NCCL és CUDA C-könyvtárak Zig FFI-kötései: ncclAllReduce, ncclBroadcast, ncclReduce, ncclAllGather, cudaMalloc, cudaMemcpy, cudaStreamCreate stb. Ezek nélkül nem lehetséges a multi-GPU kommunikáció.

---

 src/hw/accel/ – Hardveres gyorsítás

accel_interface.zig az RSFAccelerator interfészt valósítja meg, amely a Futhark-kerneleket hívja: forward, backward, trainingStep, setWeightsS/T, setSBias/TBias, setClipRange és sync metódusokkal. A FutharkArray2DF16 és PinnedMemory típusok a GPU-memória kezelését végzik.

---

cuda_bindings.zig CUDA FFI-kötések a Futhark-kontextus inicializálásához és a GPU-memória kezeléséhez.

---

fractal_lpu.zig egy Fractal Language Processing Unit szimulációja, amely fraktál-dimenzió alapú feldolgozást végez az NSIR-gráf csomópontjain.

---

futhark_bindings.zig a Futhark futtatókörnyezet Zig FFI-kötései: kontextus-inicializálás, szinkronizálás és a generált C-kód hívása.

---

futhark_kernels.fut a Futhark GPU-kerneleket tartalmazza: az RSF forward/backward lépések párhuzamos implementációja, amelyek automatikusan fordulnak CUDA/OpenCL kódra. A Futhark funkcionális párhuzamos programozási modellje garantálja a helyes párhuzamosítást.

---

main.fut a Futhark program belépési pontja, amely a kerneleket exportálja a Zig-kötések számára.

---

 src/zk/inference_trace.circom – ZK-áramkör

Ez a Circom 2.1.8 áramkör az RSF-inferencia kriptográfiai helyességét bizonyítja: RSFLayerComputation(dim) template Taylor-sorral közelíti az exp-függvényt fixpontos aritmetikával és Poseidon-hash-sel ellenőrzi a réteg-commitmentet, FullInferenceProof(8, 32, 64) az összes réteget láncolja és a bemenet/kimenet commitmenteket, a réteg-commitmenteket és a négyzetes hibát ellenőrzi, InferenceTraceWithBatch kötegelt bizonyítékokat és Merkle-fa alapú batch-root ellenőrzést végez. A main komponens FullInferenceProof(8, 32, 64) – 8 réteg, 32-dimenziós embedding, 64-bites precizitás. Ez teszi lehetővé, hogy az inferencia eredménye nyilvánosan ellenőrizhető legyen a súlyok felfedése nélkül.
JAIDE v40

Tartalomjegyzék

1. JAIDE v40 – A projekt áttekintése
2. Első lépések — Build, Configuration & Entrypoints
3. Az architektúra áttekintése – RSF, NSIR és a Processing Pipeline
4. Alapprimitívek
5. Tenzorrendszer
6. Memóriakezelés
7. I/O és Model Persistence
8. Neurális feldolgozás – RSF és OFTB
9. RSF — Reversible Scatter Flow Processor
10. OFTB — Ortogonális fraktál transzformációs blokk
11. Tokenizátor és visszakeresés
12. MGT – Morféma-vezérelt tokenizer
13. SSI – Structured Sequence Index
14. Rangsoroló – Sorozatpontozás és jelöltértékelés
15. NSIR – Kvantum-relációs gráfrendszer
16. NSIR Core – Gráfstruktúra és kvantumműveletek
17. Érvelés, hangszerelő és energiaminimalizálás
18. CREV Pipeline – Tudáskinyerés és hármaskezelés
19. Quantum Backend integráció
20. Optimalizálás és képzés
21. SFD Optimizer – Másodrendű képzés
22. Elosztott képzés
23. Cloud Training modálissal
24. Hardveres gyorsítási réteg
25. Futhark GPU kernelek
26. CUDA kötések és gyorsító interfész
27. Clash RTL Components
28. Következtetési szerver és API
29. InferenceServer – HTTP API és Request Lifecycle
30. Verified Inference Engine és ZK Proofs
31. Biztonság, biztonság és formális ellenőrzés
32. Formai igazolás és biztonsági igazolások
33. Biztonság, zavarás és C API
34. Szójegyzék

JAIDE v40 — Projekt áttekintése

A JAIDE (v40) egy nagy nyelvi modell (LLM), amely az alapoktól kezdve a Reversible Scatter Flow (RSF) paradigmára épül. A hagyományos Transformer vagy CNN architektúrákkal ellentétben a JAIDE bijektív csatolási rétegeket használ, amelyek lehetővé teszik az O(1) memória visszaterjesztését, és egy paraméter nélküli Haar-wavelet keverőblokkot, amely az OFTB néven ismert.

A rendszert nagy teljesítményű végrehajtásra tervezték a hardverek széles skáláján, a szabványos CPU-któl a több GPU-s B200-as klaszterekig és a kvantumrelációs gráfokig.

Az RSF paradigma

A JAIDE magja a Reversible Scatter Flow, amely a hagyományos önfigyelem és MLP struktúrákat kereszt-affin csatolással váltja fel.

Bijektivitás: Minden előrelépésnek van egy pontos algebrai inverze, amely biztosítja, hogy a feldolgozás során ne essenek össze az információk.
Memóriahatékonyság: Mivel a hálózat megfordítható, az aktiválásokat nem kell gyorsítótárban tárolni a visszaterjesztéshez. A rendszer a bemeneteket a helyben lévő kimenetekből rekonstruálja.
Minimalista primitívek: Az architektúra kiküszöböli a softmax, attention, ReLU és LayerNorm elemeket, kizárólag a tanult skála- és fordítástenzorokra támaszkodva.

Rendszerarchitektúra és adatfolyam

A következő diagram áthidalja a fogalmi "természetes nyelvi teret" a "kódentitástérrel", amely bemutatja, hogyan folyik át a kérés az elsődleges Zig-összetevőkön.

Kulcsfontosságú építészeti pillérek

1. Alapvető primitívek és memória
A rendszer egy egyedi Tensor rendszerre és egy speciális elosztócsomagra (Arena, Slab, Buddy) támaszkodik a memória kezeléséhez az általános célú halom többletköltsége nélkül.

2. RSF Processing Pipeline
A LayerCore a számítás alapegysége. Pontosan négy tanulható tenzorból áll: s_weight, t_weight, s_bias és t_bias. A fraktálkeverést a OFTB blokk kezeli, amely pillangó stílusú Haar-wavelet transzformációt valósít meg.

3. Tokenizálás és visszakeresés
A JAIDE a Morpheme-Guided Tokenizer-t (MGT) használja a szövegbontáshoz, a Strukturált Sequence Indexet (SSI) pedig a hatékony hasonlóságkereséshez és tudás-visszakereséshez. 4. NSIR (kvantum-relációs gráf)
A Non-linear Self-Similar Information Retrieval (NSIR) rendszer hierarchikus érvelési réteget biztosít. Kvantumlogikai kapukat (Hadamard, CNOT) integrál klasszikus aktiválással, hogy komplex kapcsolatokat modellezzen egy önhasonló gráfstruktúrán belül.

5. Hardveres gyorsítás
A számítást a Futhark GPU-kernelek (CUDA/OpenCL) gyorsítják fel az RSF áramláshoz, és a Clash az RTL hardverszintézishez.

Gyermekoldalak

A JAIDE v40 egyes összetevőinek mélyebb megismeréséhez olvassa el a következő részeket:

Elmagyarázza a Zig build rendszert, a gpu jelző engedélyezését, és a különféle végrehajtható célokat, például a jaide-inference-server és jaide-gpu.
Koncepcionális mély merülés a Reversible Scatter Flow matematikai alapjaiban, valamint az idegi mag és a kvantumrelációs gráf közötti adatáramlásban.

Első lépések — Build, Configuration & Entrypoints

Ez az oldal részletezi a JAIDE v40 rendszer összeépítési infrastruktúráját, konfigurációs beállításait és elsődleges belépési pontjait. A JAIDE a Futhark által generált C kernelekkel integrált Zig build rendszert használja a nagy teljesítményű neurális feldolgozáshoz.

Építsen rendszert és eszközláncot

A JAIDE a Zig 0.13.0 használatával készült. Az összeállítási folyamat mind a Zig-forráskódot, mind a hardvergyorsított kernelek fordítását kezeli.

A Zig építési folyamat
A build.zig szkript határozza meg az összes rendszerösszetevő fordítási folyamatát. A build kritikus része a futhark_kernels.c integrálása, amely tartalmazza a Futhark által generált C-kódot a GPU és SIMD gyorsításhoz.

Műtárgy - Fájl - Leírás
:--- - :--- - :---
jaide - src/main.zig - Az elsődleges CLI interaktív használatra, képzésre és REPL-re.
jaide-inference-server - src/inference_server_main.zig - HTTP/1.1 szerver a modell telepítéséhez.
jaide-distributed - src/main_distributed.zig - Több csomópontos edzőheveder (-Dgpu=true szükséges).
jaide-gpu - src/main_gpu.zig - Optimalizált egycsomópontos H100/A100 képzési belépési pont.

Konfigurációs jelzők létrehozása
Az összeállítási rendszer támogatja a feltételes fordítást a következő összeállítási opciókon keresztül:
GPU-gyorsítás: A -Dgpu jelző vezérli. Ha engedélyezve van, a gpu_acceleration build opciót true értékre állítja, lehetővé téve az elosztott képzést és a GPU-specifikus végrehajtható fájlokat.
Optimalizálási szintek: A szabványos Zig optimalizálási szinteket (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall) a -Doptimize támogatja.

Összeállítási folyamat
A következő diagram bemutatja, hogy a Zig build rendszer hogyan hangolja össze a Zig forrás és a Futhark C kernelek fordítását.

Rendszerbelépési pontok

A JAIDE számos speciális belépési pontot biztosít a kívánt művelettől függően (következtetés, helyi képzés vagy elosztott GPU képzés).

1. Fő végrehajtható fájl (jaide)
Az elsődleges belépési pont a src/main.zig. Kezeli a rendszer inicializálását, beleértve a RSF (Reversible Scatter Flow) processzort, a MGT tokenizert és a SSI indexet. MainConfig struktúrát használ az olyan alapértelmezett hiperparaméterek meghatározásához, mint a DEFAULT_EMBEDDING_DIM (128) és DEFAULT_RSF_LAYERS (4).

2. Következtetési szerver (jaide-inference-server)
A src/inference_server_main.zig-ben meghatározott belépési pont a InferenceServer-t ServerConfig-vel inicializálja. A CLI argumentumokat elemzi a hálózati környezet konfigurálásához:
--port: Figyelő port (alapértelmezett 8080).
--host: Összerendelési cím (alapértelmezett 0.0.0.0).
--model: A .jaide modellfájl elérési útja. 3. GPU képzés (jaide-gpu)
A src/main_gpu.zig belépési pontot az NVIDIA hardveren (pl. H100) végzett nagy teljesítményű oktatáshoz tervezték. Inicializálja a GPUCoordinator és DistributedTrainerFuthark paramétereket, és az NCCL-t használja a kommunikációs primitívekhez.

Konfiguráció és hiperparaméterek

A rendszer viselkedését a MainConfig és Config struktúrák szabályozzák a src/main.zig-ben.

Alapértelmezett hiperparaméterek
Paraméter - Alapértelmezett érték - Tartomány
:--- - :--- - :---
embedding_dim - 128 - 8 - 16 384
rsf_layers - 4 - 1 - 256
batch_size - 16 - 1 - 4,096
learning_rate - 0,001 - 1e-10 - 10,0
sequence_length - 64 - N/A

Fájlvarázsszámok
A JAIDE speciális varázsszámokat használ a bináris szerializáláshoz, hogy biztosítsa a fájl integritását:
RSF modell: 0x4A524653
MGT Tokenizer: 0x4A4D4754
Rangsoroló: 0x4A524E4B

Célok tesztelése

A build rendszer meghatározott tesztcsomagokat határoz meg, amelyek a Zig CLI-n keresztül hívhatók meg.

Parancs - Célforrás - Leírás
:--- - :--- - :---
zig build test - src/main.zig - Minden egységtesztet futtat a kódbázison keresztül.
zig build test-tensor - src/core/tensor.zig - Tenzor matematikai, SIMD és memóriaelrendezések tesztelése.
zig build test-memory - src/core/memory.zig - Érvényesíti az egyéni allokátorokat (Arena, Slab stb.).

Minden teszt össze van kapcsolva a libC és a Futhark kernelekkel, így biztosítva, hogy a hardvergyorsítási útvonalak ellenőrzésre kerüljenek a tesztciklus során.

Az architektúra áttekintése – RSF, NSIR és a feldolgozási csővezeték

Ez az oldal a JAIDE v40 architektúra fogalmi és technikai térképét tartalmazza. Leírja, hogy a Reversible Scatter Flow (RSF) neurális mag, a Non-linear Self-Similar Information Retrieval (NSIR) kvantumrelációs gráf és a támogató feldolgozási folyamat (tokenizer, optimalizáló és következtetési szerver) hogyan integrálódik egységes rendszerré.

Rendszerintegrációs térkép

A JAIDE v40 folyamat az adatokat a "természetes nyelvi térből" egy nagy dimenziójú "kód entitástérbe" helyezi át, ahol kvantumrelációs dinamikán keresztül történik az érvelés.

Adatfolyam áttekintése

1. Lenyelés: A nyers szöveget a MGT (Morpheme-Guided Tokenizer) dolgozza fel.
2. Vetítés: A tokenek Tensor primitívekké alakulnak.
3. Neurális mag: A RSFLayer bijektív transzformációkat hajt végre a LayerCore primitív használatával.
4. Relációs leképezés: A beágyazásokat a SSI (Structured Sequence Index) indexeli, és a NSIR grafikon csomópontjaira képezi le.
5. Érvelés: A ReasoningOrchestrator minimalizálja a grafikonok energiáját a ThoughtLevel hierarchiákon keresztül.
6. Következtetés: A InferenceServer ezeket a képességeket egy REST API-n keresztül teszi elérhetővé.

Összetevők kapcsolati diagramja

Az RSF neurális mag

A Reversible Scatter Flow (RSF) a JAIDE alapvető számítási paradigmája. A Transformerstől eltérően bijektív csatolási rétegekre támaszkodik, lehetővé téve az O(1) memória visszaterjesztését.

LayerCore primitív
A LayerCore az egyetlen betanítható primitív a hálózatban. Négy tenzorból áll:
s_weight / s_bias: Skála paraméterek.
t_weight / t_bias: Fordítási paraméterek.

Bijective Pipeline
Az előrelépés (forwardInPlace) és az inverz lépés (inverseInPlace) pontos algebrai inverzek, biztosítva, hogy a feldolgozás során ne essenek össze az információk. - Művelet - Logika - Memória összetettsége -
:--- - :--- - :---
Tovább - y_1 = x_1 \odot \exp(W_s x_2 + b_s) - O(1) (Helyben)
Inverz - x_1 = y_1 / \exp(W_s x_2 + b_s) - O(1) (Helyben)
Hátsótámasz - backwardFromOutputs rekonstruálja a x-t y-ből - O(1) (Nincs aktiválási gyorsítótár)

NSIR kvantumrelációs gráf

A Non-linear Self-Similar Information Retrieval (NSIR) rendszer magas szintű érvelést kezel azáltal, hogy a tudást kvantumállapotok grafikonjaként ábrázolja.

Csomópont- és éldinamika
A NSIR gráf csomópontjai egy Qubit állapotot tartalmaznak, amely valószínűségi igazságot vagy aktiválást jelent. Az éleknek van egy quality (pl. entangled, collapsed, fractal), amely meghatározza, hogy az információ hogyan áramlik a fogalmak között.

A feldolgozási csővezeték

A rendszer folyamatos áramlásként működik a nyers inputtól a strukturált érvelésig.

1. Tokenizáció (MGT)
A MGT struktúra háromszintű megközelítéssel bontja morfémákra a szöveget:
1. Speciális tokenek: [PAD], [UNK], [BOS], [EOS].
2. Morfológiai bontás: Az elő- és utótagok prioritást élveznek.
3. BPE Fallback: Byte-Pair kódolás ismeretlen szekvenciákhoz.

2. Strukturált indexelés (SSI)
A SSI (Structured Sequence Index) hídként működik a neurális beágyazások és a relációs gráf között. A Hamming-távolság hasonlóságot használja a legjobb K jelöltek érveléséhez.

3. Következtetési szerver
A InferenceServer kezeli a kérés életciklusát:
Kérés: JSON fogadása a POST /v1/inference-en keresztül.
Végrehajtás: hangszerel MGT -> RSF -> SSI -> NSIR.
Memória: ArenaAllocator-t használ kérésenként a nagy teljesítményű, szivárgásmentes működés érdekében.

Csővezeték adatfolyam-diagramja

Alapvető primitívek

Az alapprimitívek a JAIDE v40 verem legalacsonyabb szintjét képviselik, biztosítva a numerikus számítások, a memóriabiztonság és a tartós tárolás alapvető építőelemeit. Ezeket a segédprogramokat nagy teljesítményre és szigorú biztonságra tervezték, és az RSF neurális motor és az NSIR gráfrendszer alapjául szolgálnak.

Adatfolyam és entitásleképezés

A következő diagramok szemléltetik, hogy a "Természetes nyelvi tér" magas szintű fogalmai hogyan képezik le az alapvető primitíveken belüli meghatározott "kód entitásokat".

Tenzoros és numerikus tértérképezés
Memória és I/O infrastruktúra leképezés

Tenzor rendszer

A Tensor struktúra a numerikus adatok elsődleges hordozója a JAIDE-ben. Támogatja a többdimenziós alakzatokat (legfeljebb 8 dimenzióig), és írásra másolási (COW) mechanizmust használ, hogy minimalizálja a memória többletterhelését az átalakítások során.

Alak és lépések: A Shape segédprogram kezeli a méretméreteket, és kiszámítja a lépések lépéseit a nem összefüggő memóriaelérés érdekében.
Memóriaintegráció: A tenzorok inicializálhatók különféle speciális allokátorokkal, beleértve a ArenaAllocator, PoolAllocator és SlabAllocator.
Iteráció: A TensorIterator egységes módot biztosít a tenzorok bejárására, függetlenül azok mögöttes memóriaelrendezésétől vagy lépkedésétől.

Részletekért lásd: Tenzorrendszer.

Memóriakezelés

A JAIDE egyéni elosztók sorozatát használja, amelyek célja a töredezettség kiküszöbölése és determinisztikus teljesítmény biztosítása a különböző munkaterhelési mintákhoz.

Allokátor - Cél - Fájl hivatkozás
:--- - :--- - :---
Arena - Gyors, lineáris kiosztás a kérés hatókörű adatokhoz.
Slab - Fix méretű objektumok hatékony kezelése.
Pool - Szálbiztos kiosztás egységes objektumtípusokhoz.
- Buddy - Kettős erőkiosztás a külső széttagoltság csökkentése érdekében. - - A rendszer a secureZeroMemory-t is tartalmazza annak biztosítására, hogy az érzékeny adatok (például modellsúlyok vagy dekódolt tenzorok) használat után azonnal törlésre kerüljenek a RAM-ból.

A részletekért lásd: Memóriakezelés.

I/O és Model Persistence

Az I/O réteg nagy teljesítményű fájlhozzáférést és robusztus szerializációs keretrendszert biztosít a modellsúlyokhoz és a gráfállapotokhoz.

Memórialeképezés: A MMAP megvalósítás lehetővé teszi a rendszer számára, hogy a lemezen lévő nagy fájlokat bájtpufferként kezelje a memóriában, amely támogatja mind a megosztott, mind a privát leképezéseket.
Biztonságos I/O: A IoConfig szigorú korlátozásokat határoz meg a fájlméretekre és az elérési útvonalak hosszára vonatkozóan, hogy megakadályozza az erőforrás-kimerítő támadásokat.
Perzisztencia: A szerializációs réteg kezeli az olyan összetett struktúrák exportálását és importálását, mint az RSF LayerCore és NSIR SelfSimilarRelationalGraph.

A részletekért lásd: I/O és Model Persistence.

Megosztott típusok

A types.zig modul meghatározza a motorban használt primitív numerikus típusokat, különös tekintettel a fixpontos aritmetikára a platformok közötti bitdeterminizmus biztosítása érdekében.

Fixpontos aritmetika: Az olyan típusok, mint a FixedPoint16, FixedPoint32 és Fixed32_32, a túlcsordulás ellenőrzött összeadás, kivonás, szorzás és osztás módszereit biztosítják.
Hibakezelés: A központi Error enum határozza meg az alapvető primitívek által használt szabványos hibakódokat.

Tenzor rendszer

A Tensor System a JAIDE összes numerikus számításának alapvető adatszerkezete. Többdimenziós tömb absztrakciót biztosít tetszőleges lépések, másolás-írás (CoW) memóriakezelés és SIMD-gyorsított lineáris algebra támogatásával. A rendszert nagy teljesítményű neurális feldolgozásra tervezték, és egy robusztus iterátor mintán keresztül támogatja a folyamatos és nem összefüggő memóriaelrendezéseket.

1. Alapvető adatstruktúrák

A rendszer a Tensor struktúra és annak belső Shape metaadatai körül forog.

1.1 A tenzorstruktúra
A Tensor struktúra kezeli a numerikus adatok életciklusát. Egy atomi referenciaszámlálót használ a hatékony megosztás támogatására, és egy cow (írásra másolás) jelzőt, amely csak akkor váltja ki az adatok megkettőzését, ha egy megosztott tenzor módosul.

Mező - Típus - Leírás
:--- - :--- - :---
data - []align(32) f32 - Tekintse meg az aktív adatszegmenst.
base_data - []align(32) f32 - Eredeti lefoglalt memóriablokk.
shape - Shape - A méreteket és lépéseket leíró metaadatok.
refcount - usize - Atom referenciaszámláló memóriakezeléshez.
cow - bool - Jelző, amely azt jelzi, hogy a tenzor meg van-e osztva, és az írás előtt másolatot kell készíteni.

1.2 Alak- és lépésszámítás
A Shape struktúra kezeli a többdimenziós indexek lineáris memóriaeltolásokká való átalakítását. Maximum 8 dimenziót támogat. A lépések kiszámítása az inicializálás során történik, hogy lehetővé tegyék a "nézeteket" (pl. szeleteket vagy transzponálásokat) adatok másolása nélkül.

Folyamatos ellenőrzés: Egy alakzat akkor összefüggő, ha az egyes dimenziók lépései megegyeznek az összes következő méret szorzatával.
Műsorszórás: A broadcastCompatible függvény meghatározza, hogy a tenzor kibontható-e, hogy megfeleljen a cél alakzatának elemenkénti műveletek esetén.

2. Memóriakezelés és CoW

A Tensor rendszer egy írásra másolás mechanizmust valósít meg a szükségtelen kiosztások minimalizálása érdekében.1. Inicializálás: A Tensor.init lefoglalja az adatokat, a refcount 1-re inicializálódik, a cow jelző pedig false.
2. Retenció: A Tensor.retain egy atomfetch-add (@atomicRmw) segítségével növeli a referenciaszámot, és a cow jelzőt true értékre állítja.
3. Release: Tensor.release csökkenti a számlálást. Ha a szám eléri a nullát, felszabadítja a mögöttes base_data, refcount és cow jelzőt.
4. Egyidejűség: A rendszer feszültség-tesztelt a szálbiztos újraszámlálás érdekében atomműveletekkel.

3. TensorIterator és elrendezések

Nem összefüggő tenzorok esetén (pl. transzponálás vagy szelet után) a TensorIterator szabványos módot biztosít az elemek logikai sorrendben történő bejárására, függetlenül a fizikai memória elrendezésétől.

Állapot: nyomon követi a indices-t minden tengelyhez és az aktuális lineáris offset-t.
Advance Logic: A advance() módszer a legbelső dimenziótól kifelé növeli az indexeket, és előre kiszámított lépésekkel frissíti a offset-t.

4. Hardveres gyorsítás és aritmetika

A JAIDE SIMD-t (Single Instruction, Multiple Data) és többszálú tenzorműveleteket használ.

4.1 SIMD gyorsítás
A rendszer 8-as vector_width értéket határoz meg a f32 műveletekhez, és a @Vector(8, f32) értéket használja a párhuzamos aritmetikához. Ez olyan elemenkénti műveletekre vonatkozik, mint az összeadás, kivonás és méretezés.

4.2 mátrixszorzás (Matmul)
A rendszer két elsődleges matmul implementációt kínál:
1. Comptime Matmul: Speciális implementáció a Zig comptime használatával fix méretű mátrixokhoz (M, K, N), lehetővé téve a hurok kibontását és az agresszív optimalizálást.
2. Többszálú Matmul: Nagy tenzorok esetén a rendszer felosztja a munkaterhelést az elérhető CPU-magok között.

4.3 Lineáris algebra
Az alapvető aritmetikán túl a rendszer támogatja:
Determináns és inverz: elengedhetetlen az RSF (Reversible Scatter Flow) rétegekhez.
Fixpontos támogatás: Speciális célpontok esetén a rendszer Fixed32_32 és FixedPoint16/32/64 típusokat tartalmaz túlcsordulás ellenőrzött aritmetikával.

5. Integráció allokátorokkal

A Tensor rendszer allokátor-agnosztikus, de kényelmi wrappereket biztosít a JAIDE egyéni memóriakezelési alrendszerei számára:
Aréna: initWithArena az igény szerinti hatókörű tenzorokhoz.
Pool/Slab: initWithPool és initWithSlab fix méretű idegi súlyokhoz.
Buddy: initWithBuddy a kettős hatvány követelményeivel rendelkező dinamikus kiosztásokhoz.

Memóriakezelés

A JAIDE v40 memóriakezelő rendszer egyéni allokátorok és szinkronizálási primitívek átfogó készletét kínálja a nagy teljesítményű neurális feldolgozáshoz és biztonságos adatkezeléshez. Az architektúra a memóriahelyre, a szálak biztonságára és a determinisztikus erőforrás-életciklus-kezelésre helyezi a hangsúlyt speciális elosztási stratégiák és zárolásmentes struktúrák révén.

Memóriakonfiguráció és segédprogramok

A rendszer alapvető állandókat és segédfunkciókat határoz meg a memóriaigazítás és az aritmetikai biztonság érdekében.

Állandó - Érték / Fájl - Leírás
:--- - :--- - :---
PageSize - 4096 vagy 16384 - Rendszerspecifikus virtuális memória oldalméret.
CACHE_LINE_SIZE - 128 - Cél-igazítás a hamis megosztás megelőzése érdekében.
secureZeroMemory - Funkció - Felülírja a memóriát nullákkal, illékony műveletekkel, hogy megakadályozza a fordító feloldását.

Egyéni allokátorok

A JAIDE számos kiosztási stratégiát valósít meg a széttagoltság és a többletterhelés minimalizálása érdekében a különböző munkaterhelések között. 1. Aréna és ArenaAllocator
A Arena egy fix méretű, szálbiztos puffer a gyors allokációkhoz egylépéses felosztással. A ArenaAllocator ezt kiterjeszti a dinamikus pufferek listájának kezelésével, szabványos std.mem.Allocator interfészt biztosítva.

A legfontosabb funkciók:
Arena.init(allocator, size): Előzetesen lefoglal egy oldalhoz igazított puffert.
Arena.alloc(size, alignment): Szálbiztos ütés-kiosztás.
Arena.secureReset(): Az eltolás visszaállítása előtt nullázza az összes lefoglalt memóriát.

2. Födém- és medenceallokátorok
A külső töredezettség kiküszöbölése érdekében egységes objektumméretekhez tervezték.
SlabAllocator: Kezeli a memória "tábláit" egyenlő méretű slotokra osztva. free_list-t használ az elérhető helyek nyomon követésére.
PoolAllocator: Magasabb szintű burkoló, amely több födémet kezel, lehetővé téve a medence dinamikus növekedését a kereslet növekedésével.

3. Buddy és oldalelosztók
BuddyAllocator: A bináris baráti rendszert valósítja meg a két méretű blokkokhoz, egyensúlyt biztosítva a rugalmasság és a töredezettség-szabályozás között.
PageAllocator: Vékony burkolat a rendszerszintű virtuális memóriahívások körül, biztosítva, hogy minden kiosztás oldalhoz igazodjon.

4. TrackingAllocator
Memóriaszivárgások és csúcshasználat figyelésére szolgáló diagnosztikai burkoló. Becsomagolja a mögöttes Allocator-t, és számlálókat tart fenn az összes lefoglalt bájthoz és az aktív allokációkhoz.

Adatfolyam: Kiosztási kérelmek kezelése

A következő ábra azt szemlélteti, hogy egy általános kiosztási kérés hogyan kerül továbbításra a ArenaAllocator logikán keresztül.

ArenaAllocator kérésfolyama

Biztonságos és tömörített tárhely

A JAIDE speciális tárolócsomagolókat biztosít az érzékeny vagy nagyméretű adatok számára.

EncryptedStorage: Becsomagolja a memóriapuffert, és biztosítja, hogy a beleírt adatok nyugalmi állapotban titkosítva legyenek a RAM-ban. A secureZeroMemory-t használja a deinit alatt, hogy megakadályozza az érzékeny adatok kiszivárgását.
CompressedStorage: Átlátszó tömörítést valósít meg nagy tenzorokhoz vagy indexekhez. Egy belső Arena-t kezel a tömörítési algoritmusok változó méretű kimenetének kezelésére.

Szinkronizálási primitívek

A rendszer mutex alapú és zármentes struktúrákat is tartalmaz a szálak közötti kommunikációhoz.

Szerkezet - Típus - Megvalósítási részlet
:--- - :--- - :---
ThreadSafeQueue - Mutex alapú - A std.Thread.Mutex és CondVar kódot használja a pop() műveletek blokkolására.
LockFreeStack - Atomic - A std.atomic.Value és a compareAndSwap jelet használja a push/pop műveletekhez a zárolási versengés elkerülése érdekében.
VirtualMemory - Burkolat - mmap/munmap (vagy VirtualAlloc) absztrakciókat biztosít az operációs rendszer közvetlen memória-leképezéséhez.

Rendszerentitás-leképezés

Ez a diagram áthidalja a magas szintű memóriakezelési koncepciókat a src/core/memory.zig-ban szereplő konkrét kód entitásokkal.

Memóriakezelő entitástérkép

I/O és Model Persistence

Az I/O és Model Persistence réteg biztosítja az alapvető infrastruktúrát a nagy teljesítményű adathozzáféréshez és a JAIDE v40 modell ökoszisztéma strukturált szerializálásához. Ez magában foglalja az alacsony szintű memórialeképezett fájlműveleteket, az atomi írási primitíveket és a ModelFormat keretrendszert, amely a neurális súlyok, gráfszerkezetek és optimalizáló állapotok fennmaradását szabályozza.

Core I/O infrastruktúra

A rendszer egy egyedi I/O réteget használ, amelyet nagy áteresztőképességű modellbetöltésre és szálbiztos paraméterfrissítésekre terveztek. Ennek központi eleme a MMAP megvalósítás, amely oldalhoz igazított felületet biztosít az operációs rendszer virtuális memória alrendszeréhez. Memórialeképezés (MMAP)
A MMAP struktúra kezeli a fájlokkal támogatott memóriarégiókat, és támogatja a megosztott és privát leképezéseket is. Kezeli az automatikus fájlméretezést a hozzáfűzések során, és egy dedikált mutexen keresztül biztosítja a szál biztonságát.

Funkció - Megvalósítási részlet
:--- - :---
Oldaligazítás - A mem.page_size-t használja a puffer igazításához és átméretezéséhez.
Atomic Sync - Támogatja a msync-et a MSF.SYNC-vel a tartós írás érdekében.
A határok biztonsága - Ellenőrzi a actual_size ellentételezéseket, és ellenőrzi a túlcsordulást.
Biztonság - Megvalósítja a secureZeroBytes-t az érzékeny adatok törléséhez a memóriában, illékony mutatók segítségével.

Pufferelt és tartós írás
A rendszer biztosítja a DurableWriter és BufferedReader paramétereket a rendszerhívási többletterhelés minimalizálása érdekében. A kritikus konfigurációs vagy metaadat-frissítéseknél a atomicWrite segítségével biztosítható a fájl integritása azáltal, hogy egy ideiglenes fájlba ír, és atomi átnevezést hajt végre.

I/O adatfolyam és komponens interakció

Modell szerializációs keretrendszer

A ModelFormat struktúra elsődleges tárolóként működik a teljes JAIDE állapot sorosításához, beleértve az RSF neurális processzort, az MGT tokenizert és a Ranker összetevőket.

Metaadatok és mágikus fejlécek
Minden modellfájl JAIDE40\x00 mágikus fejléccel kezdődik. A metaadatokat a rendszer JSON-megtisztított karakterláncként tárolja, amely olyan architekturális hiperparamétereket tartalmaz, mint a rsf_layers, rsf_dim és mgt_vocab_size.

Az exportModel/importModel folyamat
1. Header Generation: A varázsfüzért és az aktuális verziót írja.
2. Metaadatok sorosítása: A ModelMetadata fájlt JSON-ba konvertálja, és az adatfolyamba írja.
3. Alkatrészblokkok: Minden fő komponens (RSF, MGT, Ranker) különálló blokkokra van sorozva.
4. Ellenőrző összeg ellenőrzése: SHA-256 hash-t számítanak ki az adatok között, hogy biztosítsák az integritást a importModel során.

Modellformátum elrendezés

Offset - Tartalom - Típus
:--- - :--- - :---
0x00 - MAGIC_HEADER - [8]u8
0x08 - Version - u32 (LE)
0x0C - Metadata Length - u64 (LE)
... - JSON Metadata - []u8
... - Component Data - Binary
EOF - 32 - SHA-256 Checksum - [32]u8

Speciális kitartáskezelők

Tanult beágyazások
A LearnedEmbedding struktúra a save és load metódusokon keresztül kezeli saját perzisztenciáját. Egy speciális 0x4A454D42 (JEMB) mágikus fejlécet használ, és a súlyokat kis végű f32 értékekként tárolja.

NSIR és optimalizáló állapot
NSIR Graph: Megtartja a kvantumrelációs gráfot, beleértve a csomóponti állapotokat és az élminőségeket.
SFD Optimizer: Elmenti a sztochasztikus Fisher átló állapotát, beleértve a lendületet, a sebességet és a Fisher átló tenzorokat, így biztosítva, hogy az edzés zökkenőmentesen folytatódhasson.

Perzisztencia logikai leképezés

Kivonatolási és integritási segédprogramok

A rendszer számos kivonatolási stratégiát alkalmaz a különböző teljesítmény- és biztonsági követelményekhez:
SHA-256: A kriptográfiai integritás biztosítása érdekében modellellenőrző összegekhez és NSIR topológia kivonatokhoz használják.
Blake2b256: A generateRuntimeSeed-ben használható nagy entrópiájú PRNG inicializáláshoz.
mixHash: Gyors, 64 bites, nem kriptográfiai hash, amelyet belső indexeléshez és ütközéscsökkentéshez használnak.

Neurális feldolgozás – RSF és OFTBA JAIDE v40 neurális feldolgozó motorja az Invertible Neural Networks (INN) elvén épül fel. A hagyományos előrecsatolt architektúrákkal ellentétben a feldolgozó folyamatot bijektívre tervezték, lehetővé téve a bemenetek pontos rekonstrukcióját a kimenetekről és a O(1) memória komplexitását a visszaterjesztés során. Ez a Reversible Scatter Flow (RSF) csatolórétegek és az Orthogonal Fractal Transform Block (OFTB) keverő kombinációjával érhető el.

Építészeti szinergia

A folyamat során a tanulható nemlineáris transzformációk (RSF) és a rögzített lineáris keverés (OFTB) váltakoznak. Ez a struktúra biztosítja, hogy az információ a tanult paramétereken keresztül transzformálódjon, és az információs szűk keresztmetszetek elkerülése érdekében szétszóródjon a jellemződimenzióban.

RSF — Reverzibilis Scatter Flow processzor

Az RSF az idegi mag elsődleges tanulható komponense. Olyan csatolási architektúrát használ, amelyben a bemeneti tenzor particionálva van, és az egyik felét az affin transzformációk (skálázás és fordítás) kiszámítására használják a másik felére. Ez biztosítja, hogy a transzformáció Jacobi-jele háromszög alakú legyen, így a determináns könnyen kiszámítható, a függvény pedig triviálisan invertálható.

Az RSF főbb jellemzői a következők:
In-Place Operations: Mind a forwardInPlace, mind a inverseInPlace közvetlenül a tenzormemórián működik az allokációk minimalizálása érdekében.
Memóriahatékonyság: Az aktiválások újraszámításával a visszalépés során (backwardFromOutputs használatával), a rendszer elkerüli a köztes állapotok tárolását, lehetővé téve rendkívül mély modellek betanítását korlátozott hardveren.
Szálbiztonság: A súlyokhoz és lejtőkhöz való hozzáférés a RWLock-en keresztül történik az egyidejű következtetés és edzés támogatása érdekében.

A matematikai és a GPU-gyorsítás összekapcsolásának teljes műszaki részleteiért lásd az RSF – Reversible Scatter Flow Processor című részt.

Alkatrész - Kód Entitás - Fájl
:--- - :--- - :---
Konfiguráció - RSFConfig
Core Logic - LayerCore
Gyorsító - RSFAccelerator

OFTB — Ortogonális fraktál transzformációs blokk

Az OFTB paraméter nélküli "keverő" rétegként szolgál. Pillangós Haar-wavelet transzformációt valósít meg, amely globális kommunikációt biztosít a funkciók között. Míg az RSF rétegek a komplex nemlineáris leképezések tanulására összpontosítanak, az OFTB biztosítja, hogy a tenzor minden eleme több rétegen keresztül befolyásolhassa az összes többi elemet.

A transzformációt a FRACTAL_SCALE állandó (1/\sqrt{2}) szabályozza, amely megőrzi a tenzor normáját a keverési folyamat során, hozzájárulva a numerikus stabilitáshoz mély veremekben.

SIMD Vectorized: A megvalósítás a @Vector(8, f32)-t használja az adatok 8 széles részletben történő feldolgozásához, jelentősen javítva a CPU teljesítményét.
Rögzített működés: Az RSF-től eltérően az OFTB-nek nincs súlya, ami csökkenti a modell teljes paramétereinek számát, miközben fenntartja a magas expresszivitást.

A pillangótranszformáció és a SIMD megvalósítás részleteiért lásd: OFTB – Orthogonal Fractal Transform Block.

Integráció és adatáramlás

A neurális feldolgozási réteget jellemzően magasabb szintű rendszerek (például az Inference Server vagy a Training Harness) hívják meg, amelyek az RSF és OFTB blokkok sorrendjét kezelik.1. Bemenet: A Tensor a RSF réteghez tartozik.
2. Kapcsolás: LayerCore skálát (s_weight) és fordítást (t_weight) alkalmaz a tenzor egy részhalmazára.
3. Keverés: Az eredményül kapott tenzort a rendszer átadja a OFTB.forwardInPlace-nek, amely szétszórja az értékeket a tenzor méretei között.
4. Ismétlődés: Ez a folyamat megismétlődik a RSFConfig.max_layers-ben meghatározott számú rétegre.

RSF — Reverzibilis Scatter Flow processzor

A Reversible Scatter Flow (RSF) processzor a JAIDE v40 architektúra elsődleges neurális számítási motorja. Bijektív csatoláson alapuló architektúrát valósít meg, amely pontos invertibilitást tesz lehetővé, lehetővé téve a O(1) memória visszaterjesztését a kimenetek aktiválásának rekonstruálásával. Az RSF-et nagy egyidejűségű környezetekhez tervezték, és egységes felülettel rendelkezik a CPU SIMD és a GPU Futhark gyorsításához.

Építészeti tervezés

Az RSF architektúra LayerCore blokkok sorozatából áll. Minden blokk nemlineáris transzformációt hajt végre, amely matematikailag garantáltan megfordítható. Ezt egy osztott csatolási mechanizmussal érik el, ahol a bemeneti vektort felosztják, átalakítják, majd visszaszórják a látens térbe.

LayerCore csatolási matematika

Mindegyik LayerCore négy elsődleges paramétertenzort tart fenn: s_weight (skála), t_weight (fordítás), valamint a hozzájuk tartozó torzítások s_bias és t_bias.

A y = f(x) előre transzformáció egy léptékezési és fordítási mintát követ:
1. Skálakomponens (s): s = \text{clip}(\text{matmul}(x, W_s) + b_s, \text{min}, \text{max})-ként számítva.
2. A komponens fordítása (t): t = \text{matmul}(x, W_t) + b_t-ként számítva.
3. Kapcsolás: A kimenetet a y = x \cdot \exp(s) + t állítja elő.

A x = f^{-1}(y) inverz transzformációt a következőképpen számítjuk ki:
1. x = (y - t) \cdot \exp(-s).

Mivel a s és t a x bemenet függvényei oly módon, hogy megőrizzük a jakobi struktúrát, a transzformáció bijektív.

Rendszerentitás-térkép

A következő diagram áthidalja a matematikai fogalmakat a rsf.zig és accel_interface.zig fájlokban lévő konkrét kód entitásokhoz.

RSF Entity Association
Memória és párhuzamosság

O(1) Memória visszaterjesztése
Az RSF egyik legfontosabb jellemzője a backwardFromOutputs. Ellentétben a szabványos neurális hálózatokkal, amelyeknek minden réteghez el kell tárolniuk az aktiválásokat a gradiensek kiszámításához, az RSF az egyes rétegek bemenetét úgy rekonstruálja, hogy az inverz lépést futtatja a visszalépés során. Ez O(L \cdot N)-ről O(N)-ra csökkenti az edzés memóriakomplexitását, ahol L a rétegek száma és N a dimenzió.

Szálbiztonság az RWLockon keresztül
Minden LayerCore tartalmaz egy std.Thread.RwLock-t.
Olvasászár: A forwardInPlace és inverseInPlace során szereztük be, hogy lehetővé tegye az egyidejű következtetést.
Írászár: A gradiens frissítése vagy a paraméterek szinkronizálása során szerezhető be az atomitás biztosítása érdekében.

Hardveres gyorsítás (RSFAccelerator)

Az RSF rendszer a RSFAccelerator és a Futhark által generált kerneleken keresztül integrálódik a GPU hardverével. A gyorsító felügyeli a GPU-memória és a kernel-végrehajtás életciklusát.

GPU adatfolyam
A Zig Tensor rendszer és a GPU-környezet közötti adatáramlást FutharkArray absztrakciók kezelik.GPU-gyorsítócső
Billentyűfunkciók
Funkció - Leírás - Fájl
:--- - :--- - :---
init() - Inicializálja a Futhark környezetet, beállítja a 0 eszközt, és konfigurálja a csoport/csempék méretét.
forwardFromTensor() - Magas szintű belépési pont az RSF forward pass futtatásához GPU-n.
futhark_entry_rsf_forward - C-interop hívás az optimalizált GPU kernelhez.
sync() - Szinkronizálja a GPU parancssorát a gazdagéppel.

Sorozatosítási formátum (4-es verzió)

Az RSF-modellek robusztus bináris formátumot használnak (4-es verzió), amely CRC32 ellenőrző összegeket tartalmaz minden paramétertenzorhoz az adatok integritásának biztosítása érdekében.

Sorozatosítási struktúra
1. Fejléc: SAVE_VERSION (u32), dim (u64), num_layers (u64).
2. Konfiguráció: clip_min (f32), clip_max (f32).
3. Rétegadatok: Minden réteghez:
s_weight, t_weight, s_bias, t_bias tenzorok.
Minden tenzort megelőz az alakja, és ezt követi a nyers adatok CRC32 ellenőrző összege.

Registry and Handle System

A nagyméretű modellek kezelésére az RSF rendszerleíró rendszert használ. A modelleket a RSFHandle azonosítja, amely a globális RSFRegistry usize indexe körüli típusbiztos burkolólap.

Registry: Központi tároló, amely kezeli a RSF példányok kiosztását és felosztását.
Kezelők: Megakadályozza a nyers mutató szivárgását, és lehetővé teszi a modellpéldányok biztonságos keresztirányú hivatkozását.

OFTB — Ortogonális fraktál transzformációs blokk

Az Orthogonal Fractal Transform Block (OFTB) egy paraméter nélküli, bijektív keverőréteg a JAIDE v40 neurális architektúrán belül. Pillangós Haar-wavelet transzformációt valósít meg, amelyet úgy terveztek, hogy hatékony, lineáris idejű keveredést biztosítson a rejtett dimenzióban anélkül, hogy megtanult súlyozásra lenne szüksége. Rögzített ortogonális transzformáció alkalmazásával az OFTB biztosítja a jel energiájának megőrzését (izometrikus tulajdonság), miközben megkönnyíti az információ diffúziót a tenzoron keresztül.

Építészeti szerep

A Reversible Scatter Flow (RSF) csővezeték összefüggésében az OFTB globális keverőként szolgál, amely követi a LayerCore csatolórétegek lokalizált nemlineáris transzformációit. Mivel szigorúan ortogonális és paramétermentes, hozzájárul a modell azon képességéhez, hogy összetett mintákat reprezentáljon fraktálszerű önhasonlóság révén anélkül, hogy növelné a paraméterek számát vagy a memóriaterületet a gradiens tároláshoz.

Adatfolyam integráció

Az OFTB közvetlenül a Tensor helyben tárolt adatokkal működik. A bemeneti dimenziót két felére osztja, és elforgatást alkalmaz a jellemzőtérben.

Matematikai megvalósítás

Az OFTB normalizált Haar-stílusú pillangó transzformációt valósít meg. A transzformációt a FRACTAL_SCALE konstans határozza meg, amely 1/\sqrt{2} \approx 0.7071067811865476-ra van állítva a művelet ortogonalitásának megőrzése érdekében.

Forward Transform
Két vektorfele esetén a és b:
a_{out} = (a - b) \cdot \text{scale}
b_{out} = (a + b) \cdot \text{scale}

Visszafelé transzformáció (inverz)
A művelet megfordítása visszaszaporítás vagy inverz következtetés során:
a_{in} = (a_{out} + b_{out}) \cdot \text{scale}
b_{in} = (b_{out} - a_{out}) \cdot \text{scale}

Állandó - Érték - Leírás
:--- - :--- - :---
FRACTAL_SCALE - 0.7071067811865476 - A 1/\sqrt{2} skálázási tényező, amely biztosítja az egységhatározót.
VLEN - 8 - SIMD vektor hossza a f32 műveletekhez.

Code Entity Mapping

A következő diagram leképezi az ortogonális fraktál transzformáció logikai műveleteit a oftb.zig konkrét implementációs entitásaira. SIMD vektorizálás

Az OFTB nagy áteresztőképességű feldolgozásra van optimalizálva a Zig @Vector típusával. Az implementáció 8 f32 elemet dolgoz fel iterációnként (256 bites vektorok), mielőtt a fennmaradó elemek skalárhurkába esnének vissza.

Vektorosított előrehaladás
Az előrepassz az első félidőben kivonást, a második felében pedig összeadást használ a "szórás" hatás létrehozásához.

1. Betöltés: A va és vb a tenzorszelet első és második feléből töltődik be.
2. Jelölés: A FRACTAL_SCALE a vscale vektoron át van szórva.
3. Számítás:
x1 frissítés: (va - vb) vscale .
x2 frissítés: (va + vb) vscale .

Vektorizált hátramenet
A visszalépés (a backwardInPlace és a backwardInPlaceSlice esetén) megfordítja az eredeti színátmenetek vagy bemenetek visszaállításának logikáját.

1. Betöltés: A ga és gb színátmenetek a felosztott szeletekből töltődnek be.
2. Számítás:
g1 frissítés: (va + vb) vscale .
g2 frissítés: (vb - va) vscale .

Integráció és hibakezelés

A OFTB struktúra biztonsági ellenőrzéseket biztosít annak biztosítására, hogy a tenzorméretek kompatibilisek-e az osztott keverési logikával.

Inicializálás: A init(d: usize) nem nulla dimenziót igényel, és azt állítja, hogy a méret nem okoz túlcsordulást, ha megkétszerezi (mivel a blokk 2 dim összes elemen működik) .
Ellenőrzés: A forwardInPlace és a backwardInPlace egyaránt ellenőrzi, hogy a megadott Tensor vagy szelet legalább self.dim 2 elemet tartalmaz-e.
Memória: A OFTB lényegében egy metaadattároló (dim); A deinit egyszerűen undefined értékre állítja a struktúrát, mivel nem rendelkezik kupacmemóriával.

Tokenizátor és visszakeresés

A Tokenizer és Retrieval alrendszer biztosítja a hidat a nyers természetes nyelv és az RSF mag által feldolgozott nagydimenziós vektorterek között. Kezeli a szöveg diszkrét tokenekké alakítását, ezen sorozatok hatékony tárolását egy strukturált indexben, valamint a jelöltek rangsorolását a következtetés során.

Áttekintés

A csővezeték három fő összetevőből áll:
1. MGT (Morpheme-Guided Tokenizer): A szöveget morfológiai egységek és BPE tokenek hibridjére bontja.
2. SSI (Structured Sequence Index): Nagy teljesítményű hash-fa sorozatszegmensek tárolására és visszakeresésére.
3. Ranker: Pontozási motor, amely n-gramos súlyok, diverzitás-heurisztika és Jaccard-hasonlóság alapján értékeli a szekvenciajelölteket.

Adatfolyam: Szöveg a jelöltek lekéréséhez

A következő diagram azt szemlélteti, hogy a természetes nyelv hogyan alakul át indexelt entitásokká a SSI-en belül, majd ezt követően a Ranker pontozza.

Rendszerentitás-leképezés: Nyelv az indexhez

MGT — Morféma-vezérelt tokenizátor

A MGT (Morpheme-Guided Tokenizer) háromszintű tokenizációs stratégiát valósít meg. A szabványos, csak BPE-t használó tokenizátorokkal ellentétben az MGT előnyben részesíti a morfológiai lebontást (előtagok, gyökerek és utótagok), hogy jobban kezelje az erősen ragozott nyelveket és fenntartsa a szemantikai konzisztenciát.

Three-Tier Pipeline: Először ellenőrzi a speciális tokeneket (pl. [BOS], [EOS]), majd a prefixes és suffixes táblák segítségével megpróbálja a szavakat ismert morfémákra felosztani, és végül visszatér a kódolatlan kódoláshoz. részkarakterláncok.
Horgonykövetés: A tokenizátor azonosítja a „horgonyokat” – statisztikailag jelentős tokeneket, amelyek nagy megbízhatóságú pontként szolgálnak a Ranker és SSI kereséshez.
Speciális tokenek: Támogatja a szabványos fenntartott azonosítókat: [PAD] (0), [UNK] (1), [BOS] (2) és [EOS] (3).A BPE képzési algoritmus és a szókincs megmaradásának megvalósítási részleteiért lásd: MGT – Morpheme-Guided Tokenizer.

SSI — Strukturált szekvencia index

A SSI (Structured Sequence Index) egy 64 vödörből álló hash fa, amelyet a szekvenciaszegmensek O(log N) lekérésére terveztek. Elsődleges memóriaként működik a rendszer "Kód Entitásteréhez".

Adatmodell: Az adatok tárolása Segment struktúrákban történik, amelyek token tömböket, helyzeti metaadatokat és előre kiszámított pontszámokat tartalmaznak.
Ütközések kezelése: CollisionNode láncokat használ a fagyűjtőkön belüli hash ütközések kezelésére.
Hasonlósági keresés: Támogatja a Hamming-távolság alapú hasonlósági kereséseket, hogy megtalálja a releváns kontextust még tökéletlen egyezések esetén is.
Tensor integráció: Az index 134 oszlopos Tensor elrendezésbe exportálható az RSF vagy GPU kernelek tömeges feldolgozásához.

A fakiegyenlítési és bináris szerializációs formátum részleteit lásd: SSI – Structured Sequence Index.

Kód entitás leképezés: SSI belső struktúra

Rangsorrendező – Sorozatpontozás és jelöltértékelés

A Ranker értékeli a lekért szegmensek relevanciáját egy lekérdezés vagy az aktuális környezet szempontjából. Többcélú pontozási függvényt használ, amelyet RankerConfig állandókkal kalibrálnak.

N-gramm súlyozás: Csökkenő súlyokat valósít meg különböző n-gramm hosszúságokhoz a hosszabb, pontosabb egyezések prioritása érdekében.
Sokféleség és közelség: A pontszámokat a DIVERSITY_WEIGHT (a tokenek egyedisége) és a PROXIMITY_WEIGHT (a SSI ismert horgonyaitól való távolság) alapján módosítják.
Hasonlósági mérőszámok: Kombinálja a Jaccard hasonlóságot és a token átfedést annak érdekében, hogy a lekért jelöltek szemantikailag igazodjanak a bemenethez.
Párhuzamos pontozás: Támogatja a nagy jelölthalmok többszálú kiértékelését topKHeap struktúrák használatával.

A gradiens süllyedés és a streaming rangsor segítségével történő súlykalibrációval kapcsolatos információkért lásd: Rangsoroló – Sequence Scoring and Candidate Evaluation.

MGT — Morféma-vezérelt tokenizátor

A Morpheme-Guided Tokenizer (MGT) egy nagy teljesítményű, háromszintű tokenizációs rendszer, amelyet a JAIDE v40 architektúrához terveztek. A szabványos bájtpáros kódolási (BPE) rendszerekkel ellentétben az MGT a morfológiai integritást részesíti előnyben azáltal, hogy a szavakat előtagokra, gyökökre és utótagokra bontja, mielőtt visszatérne az alszavak egyesítéséhez. Ez a megközelítés biztosítja, hogy az eredményül kapott tokenek jobban illeszkedjenek a szemantikai és nyelvtani struktúrákhoz, különösen az agglutinatív nyelvekben.

Tokenizációs csővezeték

Az MGT egy hierarchikus folyamaton keresztül dolgozza fel a bevitt szöveget, hogy a nyers karakterláncokat egész számok tokenazonosítók sorozatává alakítsa.

1. Speciális token azonosítás: A tokenizátor először lefoglalt vezérlőszekvenciákat keres, mint például a .
2. Morfológiai bontás: A rendszer megpróbálja leválasztani az ismert előtagokat és utótagokat a szavakból, hogy elkülönítse a gyökeret. Ezt a gyakori morfémák előre meghatározott listái vezérlik.
3. BPE Fallback: Ha egy szó vagy annak lebontott részei nem találhatók a szókincsben, a tokenizátor byte-Pair Encoding (BPE) összevonásokat alkalmaz a tanult prioritáspárok alapján.

Adatfolyam és megvalósítás

A MGT struktúra kezeli a szókincset és a kódoláshoz és dekódoláshoz szükséges állapotot. - Alkatrész - Kód Entitás - Szerep -
:--- - :--- - :---
Szókincs - token_to_id - Térkép a gyors karakterlánc-azonosító kereséshez.
Inverz térkép - id_to_token - Térkép az azonosítók karakterláncokká való visszafejtéséhez.
Morpheme Stores - prefixes, suffixes, roots - Speciális térképek morfológiai komponensekhez.
BPE Logic - bpe_pairs - Tárolja az egyesített prioritásokat az alszavak tokenizálásához.
Horgonyok - anchors - Nyomon követi a szekvencia-illesztéshez használt nagy jelentőségű tokeneket.

Architektúra és memória integráció

Az MGT-t úgy tervezték, hogy zökkenőmentesen működjön együtt a JAIDE egyéni memóriakezelő rendszerével, lehetővé téve annak inicializálását különböző kiosztási környezetekben (Arena, Pool vagy Buddy allokátorok).

Tokenizer inicializálási folyamata
A következő diagram bemutatja, hogyan inicializálódik a MGT, és hogyan működik együtt a core_memory primitívekkel.

Műszaki adatok

Speciális token azonosítók
Az MGT speciális azonosítókat tart fenn a vezérlési áramláshoz és a kitöltéshez, biztosítva a konzisztenciát a képzési és következtetési folyamatokon.

.
.
.
.

Morfológiai logika
A initMorphemes funkció közös nyelvi egységekkel tölti fel a belső térképeket. Például tartalmazza az angol előtagokat, mint az "un-", "re-" és "pre-", valamint a magyar morfémákat, mint a "meg-", "szét-" és különféle kisbetű-végződéseket, mint a "-ban/-ben". Ez a hibrid megközelítés lehetővé teszi, hogy a modell hatékonyabban kezelje az összetett szóalakokat, mint a szabványos részszó tokenizátorok.

Kötegelt kódolás és tenzorintegráció
Az MGT-t úgy tervezték, hogy az eredményeket közvetlenül a JAIDE Tensor rendszerbe továbbítsa. Szövegköteg kódolásakor a tokenizátor egy core_tensor.Tensor objektumot állít elő, amely tartalmazza a jogkivonat-azonosítókat, amelyeket aztán betáplálhat az RSF neurális magjába.

BPE képzés és kitartás
Az MGT-ben megvalósított BPE algoritmus egy szabványos frekvencia alapú összevonási stratégiát követ, de korlátozzák a dekompozíciós fázis során kialakított morfológiai határok.

1. Párszámlálás: A rendszer azonosítja a gyakoribb szomszédos jelzőpárokat a képzési korpuszban.
2. Szabálygenerálás: A magas frekvenciájú párok BPEMerge prioritást kapnak.
3. Szókincs tartóssága: Az eredményül kapott szókincs, beleértve a BPE-szabályokat és a morfológiai térképeket, szerializálható és deszerializálható a src/core/io.zig I/O segédprogramjaival.

Tokenizer adatszerkezet-leképezés
Ez a diagram áthidalja a koncepcionális tokenizátor komponenseket a konkrét Zig-megvalósításukkal.

SSI — Structured Sequence Index

A Structured Sequence Index (SSI) egy nagy teljesítményű, hierarchikus adatstruktúra, amelyet token sorozatok indexelésére és lekérésére terveztek. 64 vödörből álló hashfa architektúrát használ, hogy hatékony tárolási és hasonlóságkeresési lehetőségeket biztosítson, áthidalva a nyers tokenfolyamok és a strukturált relációs gráfok közötti szakadékot.

SSI Tree Architecture

Az SSI többszintű hash-faként van megvalósítva, ahol minden belső csomópont 64 lehetséges gyűjtőhelyre ágazik. Ez a struktúra lehetővé teszi a keresési terület gyors szűkítését hash előtagok alapján.

Adatmodell
Az SSI a src/index/ssi.zig-ben meghatározott három elsődleges adatstruktúrára támaszkodik: Szegmens: A tároló alapvető egysége, amely a tokenek sorozatát, annak globális pozícióját, a relevancia pontszámát és a szerkezeti igazításhoz használt horgonykivonatot tartalmazza.
Csomópont: Egy ág vagy levél a fában. Az elágazó csomópontok 64 választható gyermekmutató tömbjét tartalmazzák, míg a levélcsomópontok szegmenseket tárolnak.
CollisionNode: A levél csomópontjaihoz csatolt, összekapcsolt listastruktúra a hash ütközések kezelésére, amely biztosítja, hogy több azonos hash előtaggal rendelkező szegmens is veszteség nélkül tárolható legyen.

Strukturális állandók
Állandó - Érték - Leírás
:--- - :--- - :---
bucket_width - 6 - Szintenként felhasznált bitek száma (2^6 = 64 vödör).
bucket_count - 64 - Összes gyermek belső csomópontonként.
tensor_width - 134 - A Tensor exportálásához/importálásához használt oszlopok.
max_height - 6 - A hashfa maximális mélysége.

Keresés és visszakeresés

Az SSI támogatja mind a pontos egyezést, mind a hasonlóság alapú visszakeresést. A visszakeresési folyamat egy prioritási sort használ a "Legjobb K" legrelevánsabb találatok fenntartásához a hash hasonlóság és a szegmenspontszámok kombinációja alapján.

Hamming-távolság hasonlóság
A hasonlóságkereséshez a rendszer kiértékeli a távolságot a keresési kulcs és a tárolt szegmenskivonatok között. Ez gyakran le van töltve hardveresen gyorsított komponensekre, mint például a SSISearch.hs, amely egy Mealy állapotú gépet valósít meg a fa bejárására.

Hardveresen gyorsított keresési folyamat
A következő ábra a szoftveres keresési kérésről a hardveres keresési logikára való átmenetet szemlélteti.

Tensor export és elrendezés

Az SSI a teljes állapotát Tensor formátumba tudja exportálni neurális feldolgozás vagy perzisztencia céljából. Ez az exportálás egy adott 134 oszlopos elrendezést használ a szegmensadatok és strukturális metaadatai megjelenítésére.

134 - Oszlopelrendezés leképezése
Amikor egy Segment-t tenzorsorrá konvertálunk, az adatok a következőképpen vannak csomagolva:

1. Metaadatok (0–5. oszlop): Tartalmazza a position (alacsony32/magas32-re osztva), score (bit-cast f32) és anchor_hash.
2. Token adatok (6-133. oszlop): Legfeljebb 128 token tárolódik egymás után. Ha egy szegmensben kevesebb token van, a többi oszlop általában kitömött.

Billentyűfunkciók
low32 / high32: A segédprogramok a 64 bites kivonatokat/pozíciókat 32 bites komponensekre osztják a tenzorkompatibilitás érdekében.
joinU64: 64 bites értékeket rekonstruál két 32 bites tenzoroszlopból.
refreshHash: Újraszámítja a Merkle-stílusú hash-t egy csomópont gyermekei (ág) vagy szegmensei (levél) alapján.

Tömörítés és kiegyensúlyozás

A szegmensek beillesztésekor vagy törlésekor a fa kiegyensúlyozatlanná vagy töredezetté válhat. Az SSI megvalósítás logikát tartalmaz a szerkezeti integritás fenntartására:

1. Rekurzív inicializálás: Biztosítja, hogy az összes dinamikusan hozzárendelt Node gyermek és CollisionNode lánc megfelelően felszabaduljon a memóriaszivárgások elkerülése érdekében.
2. Kivonat frissítése: Minden beillesztés alulról felfelé haladó hash-frissítést (refreshHash) indít el, biztosítva, hogy a gyökérkivonat mindig az index aktuális állapotát tükrözze.
3. Level beillesztés: A insertIntoLeaf logikája kezeli az üres levélről a lakottra való átmenetet, beleértve a Segment adatok inicializálását is.

Adatfolyam: Sorozat az indexhez

Ez a diagram áthidalja a természetes nyelv tokenizálási folyamatát az SSI tárolási modellel.

Rangsoroló – Sorozatpontozás és jelöltértékelésA Ranker alrendszer felelős az SSI-ből (Structured Sequence Index) lekért szekvenciajelöltek végső értékeléséért és kiválasztásáért. Többlépcsős pontozási motorként működik, amely n-grammos gyakorisági súlyokat, heurisztikus diverzitásmérőket és hasonlósági mérőszámokat (Jaccard/MinHash) kombinál, hogy normalizált pontszámot hozzon létre a következtetésekhez és a képzéshez.

Építészet és alappontozási logika

A Ranker struktúra kezeli a pontozási állapotot, beleértve a tanult n-gramm súlyokat és a Locality Sensitive Hashing (LSH) paramétereit. Hídként működik a nyers token sorozatok és az SSI-ben tárolt relációs fontosság között.

Többlépcsős pontozási folyamat

Az értékelés elsődleges belépési pontja a scoreSequence, amely több összetevő súlyozott összegét valósítja meg:

1. N-gramm súlyozás: A rendszer n-grammon keresztül iterál (num_ngrams-ig), és lekéri a megfelelő szegmenspontszámokat az SSI-ből. A súlyok inicializálása harmonikus csillapítással (1/n) történik.
2. Diverzitás-heurisztikus: Az egyedi tokenek és az összes tokenek arányát méri, hogy megbüntesse az ismétlődő sorozatokat.
3. Anchor Proximity: Kiértékeli a tokenek és az ismert morfológiai horgonyok közötti távolságot az SSI-gráfon belül.
4. Normalizálás: A nyers pontszám rögzítve van, és a MAX_RAW_SCORE-hez (alapértelmezett 100,0) normalizálódik.

A jelöltek értékelési folyamata
Ez a diagram azt szemlélteti, hogyan dolgozzák fel a lekérdezési sorozatot a Ranker belső összetevőin keresztül.

Hasonlósági és aláírási mérőszámok

A nagy léptékű visszakeresés kezeléséhez a Ranker MinHash és Jaccard hasonlóságot valósít meg, hogy közelítse a szekvenciák közötti átfedést, kimerítő összehasonlítás nélkül.

Jaccard-hasonlóság: AutoHashMap segítségével valósítottuk meg a tokenkészletek egyesülése közötti metszéspont kiszámításához.
MinHash/LSH: A Ranker num_hash_functions aláírást generál minden sorozathoz. Ezeket az aláírásokat a gyors közelítő hasonlósági keresésekhez használják.
Signature Generation: A stableHash-t használja a HASH_SEED_MULTIPLIER_A és B által generált forgó vetőmaggal.
LSH-aláírások: A computeMinHashSignatures kitölti a u64 szeletét, amely a tokenszekvenciában látható minimális hash értékeket képviseli.

Hardveres gyorsítás: RankerCore

A Ranker a Clash-ben (Haskell-to-RTL) meghatározott dedikált hardverkomponensen keresztüli nagy áteresztőképességű végrehajtásra készült. A RankerCore kezeli a pontszámok összegyűjtésének és a pozíció torzítás számításának teljesítménykritikus útvonalát.

RankerCore logika
A hardveres megvalósítás egy Mealy állapotú gépet (rankerT) használ a RankRequest csomagok feldolgozásához.

Position Bias: A tokennek a szegmensben elfoglalt pozíciója alapján valósítja meg a torzítást: bias = \text{scale} / (\text{pos} + 1).
Állapotkövetés: A RankerState követi a lastQuery és stateCounter jeleket, hogy kezelje az azonos lekérdezéskivonat szekvenciális rangsorolási kérelmét.

Streaming és kalibrálás

A Ranker támogatja a hosszú sorozatok valós idejű kiértékelését egy csúszó ablak mechanizmuson keresztül.

Streaming rangsor
A rendszer egy STREAMING_BUFFER_SIZE (1024) és STREAMING_WINDOW_SIZE (512) kódot használ a bejövő tokenfolyamok feldolgozásához. Ez lehetővé teszi a rangsoroló számára, hogy fenntartsa a helyi kontextust, és pontszámokat biztosítson a tipikus SSI-szegmens hosszát meghaladó sorozatokhoz.

Súlykalibrálás a gradiens süllyedés segítségével
A ngram_weights nem statikus. A calibrateWeights függvény egy alapvető gradiens süllyedési lépést valósít meg az n-gram fontosságának beállításához egy hibajel alapján (a target_score és az aktuális predicted_score közötti különbség). - Paraméter - Érték - Leírás -
:--- - :--- - :---
LEARNING_RATE - 0,01 - Lépésméret a súlyfrissítésekhez.
DIVERSITY_WEIGHT - 0,3 - Az egyedi token elosztás jelentősége.
PROXIMITY_WEIGHT - 0,3 - A morfológiai horgony közelségének jelentősége.
BASE_SCORE_WEIGHT - 0,4 - A nyers SSI szegmens pontszám súlya.

A modell kitartása

A rangsor állapota (n-gram súlyok és LSH-paraméterek) megmarad a lemezen a következetesség megőrzése érdekében a következtetési munkamenetek között.

Exportálás: A exportToFile írja a num_ngrams, num_hash_functions, seed, valamint a teljes ngram_weights és lsh_hash_params puffereket.
Importálás: A importFromFile visszaállítja ezeket a paramétereket, és újrainicializálja a Ranker példányt.

NSIR — Kvantum-relációs gráfrendszer

A Non-linear Self-Similar Information Retrieval (NSIR) rendszer a JAIDE v40 tudásreprezentációs és érvelési gerince. A hagyományos vektoradatbázisokkal ellentétben az NSIR az információt dinamikus gráfként jeleníti meg, ahol a csomópontok kvantumállapotokkal rendelkeznek (szuperpozíció, összefonódás), az élek pedig a relációs minőséget és a fraktáldimenziót tükrözik. Ez az architektúra lehetővé teszi a rendszer számára, hogy nemlineáris érvelést hajtson végre a gráf globális "energiájának" a kvantum-inspirált optimalizálás révén történő minimalizálásával.

Rendszerarchitektúra áttekintése

Az NSIR rendszer áthidalja a szakadékot a nyers strukturálatlan adatok és a magas szintű érvelés között azáltal, hogy a kivont tripleteket egy önhasonló gráfstruktúrává alakítja.

NSIR Knowledge Flow

NSIR Core — gráfstruktúra és kvantumműveletek
Az alap adatstruktúra a SelfSimilarRelationalGraph. Node objektumokat kezel, amelyek Qubit állapotot tartalmaznak, és Edge objektumokat, amelyeket egy EdgeQuality enum határoz meg.

Kvantumállapotok: A csomópontok összetett amplitúdókat (alfa/béta) használnak az információs bizonytalanság megjelenítésére.
Szegélyminőség: A kapcsolatok átmenet a következő állapotokon keresztül: superposition, entangled, coherent, collapsed és fractal.
Topológia kivonatolás: A gráf szerkezeti integritását az állapotának SHA-256 Merkle-stílusú kivonatával tartja fenn.

A részletekért lásd: NSIR Core — Graph Structure and Quantum Operations.

Érvelő hangszerelő és energiaminimalizálás
A ReasoningOrchestrator kezeli a „gondolat” életciklusát a grafikonon belül. ThoughtLevel hierarchiában működik: local, global és meta.

Energiaképlet: Az érvelés optimalizálási problémaként van megfogalmazva, ahol a rendszer a konnektivitás és a kvantumkoherencia által meghatározott "gráfenergia" függvény minimalizálására törekszik.
Cycle Execution: A hangszerelő koordinálja a ChaosCoreKernel-et az entrópiainjektáláshoz és a ESSO-t (Entangled Stochastic Symmetry Optimizer) a szerkezeti izomorfizmusok megtalálásához.

Részletekért lásd: Oktatási hangszerelő és energiaminimalizálás.

CREV Pipeline – Tudáskinyerés
A CREVPipeline (Complex Relational Extraction and Validation) feladata az NSIR gráf feltöltése külső adatfolyamokból. A szöveget vagy a strukturált adatokat RelationalTriplet objektumokká alakítja.

Kivonás: RelationPattern illesztést használ az alanyok, predikátumok és objektumok azonosítására.
Ellenőrzés: Minden hármashoz rendelnek egy anomália pontszámot és konzisztencia-ellenőrzést, mielőtt elkötelezik magukat a KnowledgeGraphIndex-re.
Kvantumleképezés: A kivonásból származó megbízhatósági pontszámok közvetlenül a csomópont Qubit állapotában lévő komplex amplitúdókra vannak leképezve.

A részletekért lásd: CREV Pipeline — Knowledge Extraction and Triplet Management.

Quantum Backend integráció
Az NSIR támogatja a szimulált és a hardveresen gyorsított kvantumműveleteket is.- Hardver: Integráció az IBM Quantum programmal a ibm_quantum.zig kliensen keresztül, amely támogatja az OpenQASM-feladatok beküldését olyan háttérrendszerekre, mint a ibm_brisbane.
Szimuláció: A RelationalQuantumLogic motor biztosítja a kapuk (Hadamard, Pauli-X/Y/Z) és a mérés által vezérelt állapotösszeomlás helyi szimulációját.
Adapter: A QuantumTaskAdapter a gráfalapú összefonódási kérelmeket QuantumCircuit kötegekre fordítja.

A részletekért lásd: Quantum háttérrendszer integráció.

Code Entity Mapping

Ez a diagram leképezi a magas szintű NSIR-koncepciókat a kódbázison belüli konkrét megvalósítási struktúráikra és fájljaikra.

A megvalósítási térkép logikája

NSIR Core — Gráfstruktúra és kvantumműveletek

A Non-linear Self-Similar Information Retrieval (NSIR) rendszer a JAIDE relációs gerince. Középpontjában a SelfSimilarRelationalGraph található, egy nagy dimenziós gráfstruktúra, ahol a csomópontok az információs entitásokat, az élek pedig a kvantumkorrelált kapcsolatokat képviselik. A klasszikus gráfokkal ellentétben az NSIR komplex értékű amplitúdókat (Qubit) használ a tudás állapotának megjelenítésére, lehetővé téve az információk szuperpozícióját és összefonódását.

1. Kvantumállapot-ábrázolás

Az NSIR kvantumprimitívek felhasználásával a csomóponti állapotokat és a kapcsolatok erősségét reprezentálja. A gráf minden csomópontja tartalmaz egy Qubit-t, amely az aktiválási állapotát jelzi, míg az élek kvantumkorrelációkat tartanak fenn.

Qubit és QuantumState
A Qubit struktúra két összetett amplitúdót tárol (a és b). A rendszer biztosítja az állapot normalizálását úgy, hogy a - a - ^2 + - b - ^2 = 1, ahol a - a - ^2 a csomópont 0 (inaktív/hamis) állapotának valószínűségét, a - b - ^2 pedig az 1. állapotot (aktív/igaz) jelenti.

A QuantumState struktúra kibővíti ezt a bonyolultabb logikai műveletekhez, a entanglement_degree és a phase követéséhez.

EdgeQuality
A csomópontok közötti kapcsolatokat a EdgeQuality listán keresztül a "koherencia" szintjük szerint osztályozzák:

Enum Value - Leírás
:--- - :---
superposition - A kapcsolat több potenciális állapotban létezik.
entangled - Az egyik csomópont állapota elválaszthatatlanul kapcsolódik a másikhoz.
coherent - Stabil, fázishoz igazodó kapcsolat.
collapsed - Határozott, klasszikus kapcsolat (a mérés eredménye).
fractal - Önhasonló kapcsolat különböző skálákon.

2. Gráf életciklusa és topológiája

A SelfSimilarRelationalGraph kezeli a tudáscsomópontok életciklusát és azok összekapcsolását. Támogatja a szabványos CRUD műveleteket a kvantumspecifikus műveletek, például az összefonódás mellett.

Csomópont és él életciklusa
1. addNode: Node-t hoz létre egyedi azonosítóval, társított adatokkal és egy kezdeti Qubit állapottal.
2. addEdge: Kapcsolatot hoz létre a forrás és a célcsomópont között, hozzárendelve a weight, quantum_correlation és fractal_dimension értékeket.
3. removeNode: Törli a csomópontot és az összes incidens élt, biztosítva a memória visszanyerését a csomópont belső allokátorán keresztül.

Topológia integritása (SHA-256 Merkle Hash)
A tudásbázis integritásának biztosítása érdekében a gráf egy computeTopologyHash függvényt valósít meg. Ez a teljes gráfstruktúra SHA-256 hash-jét generálja úgy, hogy a csomópontokon és az éleken determinisztikus sorrendben iterál, hatékonyan létrehozva az aktuális állapot Merkle-stílusú ujjlenyomatát.

Grafikon logikai adatfolyam:

3. Kvantumműveletek: Összefonódás és mérés

Az NSIR megkönnyíti az érvelést a kvantumkapukon és az állapotösszeomláson keresztül. entangleNodes: Összekapcsol két csomópontot úgy, hogy Qubit állapotaik korrelációba kerüljenek. Ez megjelenik a Edge quantum_correlation mezőjében.
mérés: Összecsukja egy csomópont Qubit-jét állapotok szuperpozíciójából klasszikus bitté (0 vagy 1) az amplitúdói által meghatározott valószínűségi eloszlás alapján. Ez a művelet visszafordíthatatlan, és átterjed a gráfon, és potenciálisan összeomolhatja az összegabalyodott szomszédokat.

Logikai kapuk
A rendszer számos kvantumkaput támogat a LogicGate segítségével:
Single-Qubit: HADAMARD, PAULI_X, PHASE, FRACTAL_TRANSFORM.
Multi-Qubit: CNOT, TOFFOLI, RELATIONAL_AND, RELATIONAL_XOR.

4. Memóriastratégia és teljesítmény

Az NSIR magot nagy áteresztőképességű relációs feldolgozásra tervezték, és a core_memory modulon keresztül több memóriakiosztási stratégiát is támogat.

Stratégia - Végrehajtás - Használati eset
:--- - :--- - :---
Aréna - std.heap.ArenaAllocator - Rövid életű érvelési ciklusok, ahol a teljes grafikont el kell vetni.
Medence - PoolAllocator - Fix méretű Node és Edge kiosztás a töredezettség minimalizálása érdekében.
Haver - BuddyAllocator - Változó méretű metaadatok és adatpuffer-lefoglalások.

Adatexportálás
A gráf segédprogramokat biztosít a relációs tér és a neurális (RSF) tér áthidalására:
exportNodeEmbeddings: A Qubit csomópont állapotait és metaadatait Tensor-má alakítja át a neurális feldolgozáshoz.
exportAdjacencyMatrix: A gráf súlyozott szomszédsági mátrixát állítja elő, amelyet gyakran használnak globális topológia elemzéshez.

Tenzorleképezés kapcsolata:

5. Időbeli dinamika és jelterjedés

A grafikon nem statikus; támogatja az időbeli verziókezelést és az aktív jelterjedést.

Időbeli grafikon: A TemporalNode és EdgeVersion struktúrák lehetővé teszik, hogy a grafikon megőrizze az állapotváltozások előzményeit, nanoszekundumos időbélyegekkel indexelve.
Jelterjedés: A SignalPropagationEngine szimulálja, hogy az aktiválások (jelek) hogyan áramlanak át a grafikonon. A jeleknek amplitude, phase és frequency van, áramlásukat pedig a EdgeQuality és weight befolyásolja.

Érvelő hangszerelő és energiaminimalizálás

A ReasoningOrchestrator az NSIR gráfrendszer központi koordinációs motorja. Kezeli a SelfSimilarRelationalGraph iteratív finomítását a helyi csomópont-perturbációk, a globális szimmetriaérzékelés és a metaszintű fraktál-újraegyensúlyozás kiegyensúlyozásával. A rendszer az Energiaminimalizálás elvén működik, ahol a gráf "energiája" a relációs tudásbázison belüli ellentmondást vagy instabilitást jelenti.

Érvelési hierarchia: Gondolati szintek

A hangszerelő az érvelést a ThoughtLevel enum által meghatározott háromszintű hierarchiába rendezi. Az egyes szintek a grafikon szerkezetének különböző részletességeit célozzák meg:

Szint - Hatály - Elsődleges művelet
:--- - :--- - :---
local - Egyedi csomópontok/élek - perturbLocalNodes, updateLocalEdges
global - Grafikon topológia - esso.optimize, transformNodes
meta - Szerkezeti integritás - rebalanceFractalTree, chaos_kernel.step

Az érvelési fázis életciklusa
Minden gondolkodási ciklus egy ReasoningPhase-be van zárva. Egy fázis követi saját energiadeltáját, és egy konfigurálható küszöb alapján határozza meg a konvergenciát.

Energiaminimalizálás és konvergencia

A hangszerelő célja egy olyan "alapállapot" elérése, ahol a relációs gráf belsőleg konzisztens. Ezt a kvantumállapot-stabilitást és a relációs minőséget ötvöző gráf energiaképlet segítségével mérik. Grafikon energia számítás
A calculateGraphEnergy függvény két elsődleges forrásból származó energiát aggregálja:
1. Csomópontpotenciál: Az egyes csomópontok Qubit állapota alapján.
2. Élfeszesség: A EdgeQuality-ből és a csatlakoztatott csomópontok közötti összefonódásból származik.

Konvergencia logika
A konvergenciát a hasConverged-ben az energia relatív változásának kiszámításával határozzuk meg:
\Delta E = \frac{ - E_{current} - E_{previous} - }{\max( - E_{previous} - , 1.0)}
Ha \Delta E < convergence\_threshold, a fázis véget ér.

Megvalósítási folyamat

A hangszerelő integrálja a EntangledStochasticSymmetryOptimizer (ESSO) és a ChaosCoreKernel-t, hogy a grafikont a stabilitás felé terelje.

Rendszerintegrációs diagram
Ez a diagram leképezi a logikai érvelési folyamatot a végrehajtásért felelős konkrét kód entitásokhoz és fájlokhoz.

Billentyűhangosítási funkciók

1. Szimmetriaérzékelés az ESSO-n keresztül
A hangszerelő meghívja a EntangledStochasticSymmetryOptimizer-t, hogy ismétlődő mintákat vagy szerkezeti szimmetriákat keressen a gráfban. Az ESSO SymmetryGroup típusokat (visszaverődés, elforgatás, fordítás) használ a redundáns információk vagy a „kanonikus” relációs struktúrák azonosítására.

2. Helyi zavarok
perturbLocalNodes: Véletlenszerűen beállítja a csomópontok egy részhalmazának kvantumamplitúdóját, hogy elkerülje a helyi energiaminimumot.
updateLocalEdges: A EdgeQuality finomítása a forrás- és célcsomópontok aktuális állapota alapján.

3. ChaosCoreKernel ciklusok
A ChaosCoreKernel kezeli az alapul szolgáló memóriát és a feladatterhelést. Egy gondolkodási ciklus során a hangszerelő elindítja a chaos_kernel.step() parancsot, hogy végrehajtsa:
Memóriakiegyenlítés: MemoryBlock entitások átcsoportosítása magok között.
Elakadások tisztítása: Elévült vagy alacsony prioritású összefonódások eltávolítása a memóriablokkok között.

Adatfolyam: A grafikon állapotának indoklása

A következő diagram bemutatja, hogy a magas szintű érvelési utasítások hogyan alakulnak át az NSIR gráfprimitívek módosításaiba.

Orchestrator Statisztika

A OrchestratorStatistics struktúra telemetriát biztosít az érvelési folyamathoz:

best_energy_achieved: A legalacsonyabb energiájú állapot az összes fázisban.
average_convergence_time: A konvergenciaküszöb eléréséhez szükséges nanoszekundumok mozgóátlaga.
patterns_discovered: Az ESSO által azonosított és a ReasoningPhase-ben rögzített SymmetryPattern objektumok teljes száma.

A recordPhase függvény minden ThoughtLevel végrehajtás végén frissíti ezeket a mutatókat.

CREV Pipeline — Tudáskinyerés és hármaskezelés

A CREV (Categorical Relational Extraction and Validation) Pipeline egy ötlépcsős feldolgozási és finomító motor, amelyet arra terveztek, hogy a strukturálatlan szöveget, strukturált adatfolyamokat és képmetaadatokat nagy pontosságú tudásgráfokká alakítsa. Elsődleges hídként szolgál a nyers bemenet és a NSIR (Non-linear Self-Similar Information Retrieval) gráfrendszer között, biztosítva, hogy az összes kivont információ konzisztenciája érvényesüljön, és kvantumrelációs állapotba kerüljön.

Pipeline Architecture and Data Flow

A CREVPipeline levezényli az információ életciklusát a kezdeti beviteltől a KnowledgeGraphIndex-ban történő végső megjelenítésig. Szakaszos megközelítést alkalmaz a komplexitás kezelésére és az adatok integritásának biztosítására. A kitermelés öt szakasza
A folyamat szigorú lineáris haladást követ, amelyet a ExtractionStage enum határoz meg:
1. Tokenizáció: Kezdeti adatfolyam-feldolgozás és morféma-vezérelt szegmentálás.
2. Triplett kivonás: Az alany-reláció-objektum minták azonosítása.
3. Ellenőrzés: Anomália pontozása és konzisztencia ellenőrzése a meglévő tudással.
4. Integráció: Konfliktusfeloldás és összevonás a SelfSimilarRelationalGraph-vel.
5. Indexelés: Optimalizálás a KnowledgeGraphIndex-en keresztüli lekérdezéshez.

Relációs Triplet Management

A CREV-folyamatban a tudás alapvető egysége a RelationalTriplet. A szabványos RDF-hármasokkal ellentétben ezek a struktúrák nagy dimenziós metaadatokat hordoznak, beleértve a megbízhatósági pontszámokat és az időbeli horgonyokat.

Adatstruktúra: RelationalTriplet
A RelationalTriplet a következőkből áll:
Alapelemek: subject, relation és object ([]u8 néven tárolva).
Magabiztosság: A f64 érték 0.0 és 1.0 között van.
Identity Hash: A hashTripletIdentity-en keresztül generált SHA-256 hash, amely megakadályozza az azonos szemantikai viszonyok ismétlődő feldolgozását.
Metaadatok: A StringHashMap(.

Kódentitás-leképezés: Triplet Identity

Érvényesítés és konfliktusmegoldás

Mielőtt egy hármast integrálnánk az NSIR gráfba, át kell mennie a validation szakaszon. Ez két elsődleges ellenőrzést foglal magában:
1. Anomália pontozás: Az új hármas összehasonlítása a ChaosCoreKernel értékkel annak megállapítására, hogy a reláció statisztikailag valószínűtlen élt jelent-e.
2. Konzisztencia-ellenőrzés: Annak ellenőrzése, hogy az új információ ellentmond-e a KnowledgeGraphIndex-ben már szereplő nagy megbízhatóságú hármasoknak.

Kvantum-relációs leképezés
A integration szakasz során a RelationalTriplet confidence-je a SelfSimilarRelationalGraph-on belüli komplex amplitúdóhoz van leképezve. Ez lehetővé teszi a rendszer számára, hogy a bizonytalanságot állapotok kvantum-szuperpozíciójaként kezelje.
Magas megbízhatóság: Összeomlott állapotot eredményez magas EdgeQuality értékkel.
Alacsony megbízhatóságú: Fenntartja a magas "energiás" állapotot, a ChaosCoreKernel gyakori zavarásának kitéve.

KnowledgeGraphIndex (háromtengelyes indexelés)

A KnowledgeGraphIndex O(1) vagy O(log N) keresést biztosít a hármasok számára a három tengely (Tárgy, Reláció vagy Objektum) bármelyike alapján. Beágyazott StringHashMap struktúrával valósul meg.

Alkatrész - Végrehajtás - Cél
:--- - :--- - :---
Elsődleges index - StringHashMap(ArrayList(RelationalTriplet)) - Egy alanyt leképez az összes kapcsolódó kapcsolatára.
Relációs index - StringHashMap(ArrayList(RelationalTriplet)) - Csoportosítja az összes hármast, amelyek egy adott relációtípuson osztoznak (pl. "is_part_of").
Objektumindex - StringHashMap(ArrayList(RelationalTriplet)) - Fordított keresés objektumról alanyra.

ChaosCoreKernel integráció
A ChaosCoreKernel „újraellenőrzési” ciklusok indításával kölcsönhatásba lép a folyamattal. Ha a globális gráf energiája meghalad egy bizonyos küszöböt, a kernel arra kényszeríti a CREVPipeline-t, hogy újraértékelje az alacsony megbízhatósági pontszámú hármasokat, ami potenciálisan a "zajos" tudás megnyirbálásához vezethet.

A műszaki megvalósítás részletei

Triplet Hashing
A rendszer különbséget tesz az Identity Hashes (S-R-O karakterláncok alapján) és a Mezőkivonatok (amelyek magukban foglalják a bizalmat és a kivonási időt) között.

hashTripletIdentity: A deduplikációhoz használatos.
hashTripletFields: Adott kibontási események auditálására és verziószámítására szolgál. Memóriakezelés
A CREVPipeline kérésenkénti ArenaAllocator-t használ a kinyerési szakaszokhoz, de a RelationalTriplet adatokat egy hosszú élettartamú SlabAllocator vagy Pool formátumba lépteti elő, ha integrálva van a KnowledgeGraphIndex-be.

Quantum Backend integráció

A Quantum Backend Integration réteg interfészt biztosít a JAIDE NSIR gráfrendszer és a kvantumszámítási erőforrások között. Támogatja mind a nagy pontosságú szimulációt a RelationalQuantumLogic motoron keresztül, mind a fizikai hardveres végrehajtást az IBM Quantum felhőszolgáltatáson keresztül. Ez az alrendszer felelős az erősen összefonódott részgráfok azonosításáért, a relációs logika OpenQASM-be fordításáért és a kvantumjobok életciklusának kezeléséért.

Rendszerarchitektúra

Az integráció többszintű absztrakcióként épül fel, kezdve az alacsony szintű kapu logikától a magas szintű gráffeladat adaptációig.

Kvantumintegrációs folyamat
Ez a diagram azt szemlélteti, hogy a QuantumTaskAdapter hogyan hidalja át a "természetes nyelvi teret" (amelyet az NSIR gráfban relációs hármasok képviselnek) a kvantumhardver és szimulátorok "kódentitásterével".

IBM Quantum Client

A IBMQuantumClient kezeli a hitelesített kommunikációt az IBM Quantum szolgáltatásokkal felhő erőforrásnevek (CRN) és API tokenek segítségével. Alapértelmezés szerint a ibm_brisbane háttérrendszert célozza meg, és OpenQASM-ben formázott feladatokat küld el.

Főbb összetevők
Hitelesítés: A IBM_QUANTUM_CRN környezeti változóból vagy manuális felülírásból beolvasott Bearer tokent és CRN-t (Cloud Resource Name) használ.
Job Submission: A submitJob függvény az OpenQASM-karakterláncokat JSON-adattömbbé szerializálja, 1024 felvételes alapértelmezett konfigurációval.
Result Retrieval: Lekérdezi az IBM Cloud API-t a feladat állapotáról, és lekéri a végső mérési bitsztringeket.

Háttér hardverspecifikációi
A rendszer karbantartja a különféle IBM architektúrák (Heron, Eagle, Falcon, Osprey, Condor) hardverspecifikációinak nyilvántartását, hogy segítse a hibamodellezést és a qubit allokációt.

Háttértípus - Qubit Count - T1 átlag (ns) - Kiolvasási hiba átlaga
:--- - :--- - :--- - :---
Gém - 133 - 350 000,0 - 0,008
Sas - 127 - 200 000,0 - 0,015
Sólyom - 27 - 100 000,0 - 0,020
Szimulátor - 32 - N/A - 0,001

Relációs kvantumlogika (szimulációs motor)

A RelationalQuantumLogic motor kvantumáramkörök helyi szimulációját biztosítja, kifejezetten relációs műveletekre optimalizálva. QuantumState objektumokat kezel, amelyek komplex amplitúdókat, fázisokat és összefonódási fokokat követnek nyomon.

Logikai kapuk
A motor támogatja a szabványos kvantumkapukat és a speciális relációs kapukat, amelyeket a gráfok érvelésére használnak:
Normál: HADAMARD, PAULI_X/Y/Z, CNOT, TOFFOLI.
Relációs: RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_NOT, RELATIONAL_XOR.
Fraktál: FRACTAL_TRANSFORM az önhasonló információk skálázására szolgál.

Quantum State Implementation
A QuantumState-t két komplex amplitúdó képviseli (a - 0\rangle és - 1\rangle alapállapotokhoz).
Normalizálás: Biztosítja a - \alpha - ^2 + - \beta - ^2 = 1 teljes valószínűséget.
Mérés: Az állapot valószínűségi összeomlása prob0() és prob1() számítások alapján.

Quantum Task Adapter

A QuantumTaskAdapter az NSIR gráf hangszerelőjeként működik. Azonosítja azokat a részgráfokat, amelyeknek az összefonódásuk és a fraktáldimenziójuk alapján előnyös lenne a kvantumfeldolgozás. Részgráf azonosítási logika
Az adapter a gráf élein keresztül iterál, és a csomópontokat QuantumSubgraph-be csoportosítja, ha meghaladják a meghatározott küszöbértékeket:
1. Összefonódási küszöb: Alapértelmezett 0.5.
2. Fraktál küszöb: Alapértelmezés 1.5.

Adatfolyam: Végrehajtás az eredményig
Amikor egy feladatot a executeTask-en keresztül hajtanak végre, az adapter:
1. A QuantumSubgraph-t LogicGate műveletek sorozatává alakítja.
2. Elküldi a local_simulator vagy a quantum_client címre.
3. A kimenetet egy QuantumTaskResult-be csomagolja, amely összetett amplitúdókat és végrehajtási statisztikákat tartalmaz.

Konfiguráció és konstansok

A QuantumConfig struktúra meghatározza a működési korlátokat mind a szimulációs, mind a hardveres háttérprogramok számára.

Állandó - Érték - Cél
:--- - :--- - :---
MAX_QUBITS_SIMULATION - 20 - Korlátozza a memóriahasználatot a helyi állapotvektorokhoz
HARDWARE_MAX_SHOTS - 100 000 - Maximális mintavételi gyakoriság IBM hardver esetén
SIMULATOR_QUBITS - 32 - Maximális címezhető qubit a szimulátorban
DEFAULT_SHOTS - 4000 - Alapértelmezett mintavétel a statisztikai konvergenciához
POLL_INTERVAL_MS - 100 - Várakozási idő a munkaállapot-ellenőrzések között

Optimalizálás és képzés

Ez a rész áttekintést nyújt a JAIDE v40 képzési csomagról, amely egy nagy teljesítményű másodrendű optimalizálót, egy elosztott képzési kábelt a több GPU-s skálázáshoz és a Modalon keresztüli felhőalapú hangszerelést tartalmaz. A rendszert úgy tervezték, hogy kihasználja a B200 GPU architektúrákat és az NCCL-t a nagy áteresztőképességű modellek konvergenciájához.

Képzési verem áttekintése

A JAIDE képzési folyamat három fő pillérre épül:
1. SFD Optimizer: Kifinomult, másodrendű optimalizáló motor, amely a Stochastic Fisher Diagonal frissítéseket és a SophiaSOAP előkondicionálást valósítja meg.
2. Elosztott kábelköteg: Súly-delta átlagoló rendszer, amely több GPU-munkást koordinál az NCCL kollektív műveletei segítségével.
3. Cloud Orchestration: Python-alapú szkriptek Modalhoz, amelyek automatizálják a környezet kiépítését, az adatkészletek feldolgozását és a több csomópontos végrehajtást.

Rendszerarchitektúra diagram

A következő diagram a képzési komponensek és a mögöttes hardveres absztrakció közötti kapcsolatot szemlélteti.

Képzési rendszer interakció
Alkatrészek lebontása

SFD optimalizáló – Másodrendű képzés
A Stochastic Fisher Diagonal (SFD) optimalizáló a modellkonvergencia elsődleges motorja. A szabványos elsőrendű módszerekkel (SGD/Adam) ellentétben az SFD másodrendű információkat használ fel, hogy hatékonyabban navigáljon a veszteségterületen.

Főbb jellemzők:
SophiaSOAP: KFAC előkondicionálást és Hutchinson Hessian becslést valósít meg a másodrendű görbületkorrekcióhoz.
Mixed Precision: A MixedPrecisionTrainer támogatja az FP32-tól az FP4-ig terjedő kvantálási szinteket a memóriahatékony képzés érdekében nagy modelleken.
Memóriakezelés: Speciális B200MemoryManager (TMEM) támogatás a nagy sávszélességű memóriahasználathoz.

A részletekért lásd: SFD Optimizer – Másodrendű képzés.

Elosztott képzés
Az elosztott kábelköteg lehetővé teszi a JAIDE számára, hogy több GPU-n és csomóponton skálázzon. Súly-delta átlagolási mintát követ a fürt konzisztenciájának megőrzése érdekében. Főbb összetevők:
DistributedTrainerFuthark: A fő belépési pont a GPU-gyorsított képzéshez, az adatkészlet-particionálás kezeléséhez és a helyi kötegelt feldolgozáshoz.
GPUCoordinator: Kezeli az NCCL életciklust, olyan primitíveket biztosítva, mint a allReduce, broadcast és barrier a modellállapotok szinkronizálásához.
Adatkészlet-particionálás: Automatikusan kezeli a JSONL adatfolyam felosztását, hogy minden dolgozó egyedi mintákat dolgozzon fel.

A részletekért lásd: Elosztott képzés.

Cloud Training modálissal
A 8×B200-as fürtök telepítésének leegyszerűsítése érdekében a JAIDE átfogó, modális alapú felhővermet tartalmaz.

Munkafolyamat:
Image Build: Egyéni Ubuntu-alapú kép, amely tartalmazza a Zig 0.13.0, Futhark és CUDA 12.4 fájlokat.
Adatkészlet feldolgozása: Automatikusan letölti és átalakítja a finephrase adatkészletet képzésre kész JSONL formátummá.
Runtime Compilation: Ha hiányoznak az előre elkészített binárisok, a szkript a Futhark kernelekből és a Zig futtatható fájlból álló tartalék buildet indít el a távoli dolgozón.

A részletekért lásd: Cloud Training with Modal.

Képzési logikai folyamat

A következő diagram leképezi a logikai folyamatot a Python hangszerelési rétegtől a Zig-alapú képzési ciklusig.

Kód entitás leképezés: hangszerelés a végrehajtásig
Hiperparaméter-konfiguráció

A képzést a TrainerConfig és TrainingParameters struktúrák szabályozzák, amelyek meghatározzák a tanulási sebességet, a lendületet és az építészeti korlátokat.

Paraméter - Típus - Alapértelmezett - Leírás
:--- - :--- - :--- - :---
learning_rate - f32 - 0.001 - A frissítések alaplépése
momentum - f32 - 0.0 - Lendületi tényező SFD
max_line_size - usize - 10MB - Max puffer az adatkészlet JSONL-soraihoz
checkpoint_version - u32 - 4 - Verziózás az RSF-modell tartósságához

SFD Optimizer — Másodrendű képzés

A Stochastic Fisher Diagonal (SFD) optimalizáló a JAIDE elsődleges másodrendű oktatómotorja. Egyesíti a természetes gradiens süllyedés elemeit a Fisher információs előkondicionáláson keresztül fejlett varianciacsökkentéssel és vegyes precíziós technikákkal, hogy lehetővé tegye a nagy dimenziós RSF architektúrák hatékony betanítását.

Alapvető megvalósítás és adatáramlás

Az SFD-optimalizáló a modellparaméterek frissítési életciklusát az elsőrendű lendület (sebesség) és a másodrendű görbületi becslések (Fisher-átló) követésével kezeli. A Sophia és a SOAP által ihletett előre kondicionált gradiens megközelítést alkalmazza, hogy a tanulási sebességet a veszteségi felület helyi geometriája alapján igazítsa.

SFD államkezelés
A SFD struktúra fenntartja az optimalizáló állapotát, beleértve a hiperparamétereket és az impulzus- és görbületi puffereket.

Alkatrész - Kód Entitás - Leírás
:--- - :--- - :---
Lendület - velocity - Nyomon követi a színátmenetek exponenciálisan súlyozott mozgóátlagát.
Fisher Diagonal - fisher_diag - Nyomon követi a négyzetes színátmenetek mozgóátlagát (vagy Hess-átlókat).
Előkondicionálás - SophiaSOAP - KFAC-stílusú előkondicionálást és Hutchinson Hessian becslést valósít meg.
Szóráscsökkentés - MARS - Csökkenti a gradiens zajt sztochasztikus beállításokban.

Rendszeradatfolyam
A következő diagram bemutatja a gradiensek áramlását az SFD optimalizálási folyamaton keresztül, a nyers visszaterjesztési kimenetektől a kvantált paraméterfrissítésekig.

Optimalizáló adatfolyama: Gradiens a paraméterekig

Kulcsfontosságú összetevők 1. Sztochasztikus Fisher-átló (SFD)
A SFD osztály a képzési kör központi koordinátora. A frissítési szabályt hajtja végre:
\theta_{t+1} = \theta_t - \eta \cdot \text{Preconditioner}(m_t, \hat{F}_t)
ahol m_t az impulzus és \hat{F}_t az átlós Fisher-becslés.

Inicializálás: init(allocator, params, options) beállítja a sebességet és a Fisher puffereket.
Frissítési lépés: A step() a számított gradienseket alkalmazza a paraméterekre.

2. SophiaSOAP és KFAC előkondicionálás
Az optimalizáló a Sophia (másodrendű sztochasztikus optimalizálás) és a SOAP (optimális bármilyen sorrendű előkondicionálású sampon) hibridjét valósítja meg. A Hutchinson-módszert használja a Hess-diagonális becslésére explicit mátrixszámítás nélkül.

Hutchinson becslés: Rademacher zajvektorokat használ a diag(H) közelítésére.
KFAC előkondicionálás: A Fisher információs mátrixot blokk-átlós mátrixként közelíti meg a hatékony inverzió érdekében.

3. MixedPrecisionTrainer
A nagy teljesítményű hardverek, például az NVIDIA B200 támogatásához az SFD többféle precíziós formátumot támogat. A quantizeValue függvény kezeli a f32 színátmenetek leképezését kisebb pontosságú ábrázolásokra.

Precíziós - A megvalósítás részletei - Tartomány
:--- - :--- - :---
4. keretprogram - 3 bites mantissza, 1 bites jel, egyéni szintek - [-6,0, 6,0]
8. keretprogram - E4M3/E5M2 stílusú kvantálás - [-448,0, 448,0]
FP16 - Szabványos félpontos közelítés - [-65504, 65504]

Bayesi optimalizálás és LR ütemezés

A képzési folyamatot egy BayesianOptimizer szabályozza, amely egy GaussianProcess helyettesítő modell segítségével hangolja a hiperparamétereket (például a tanulási sebességet és a súlycsökkenést).

Bayesi hangolócső
LRSütemező
A LRScheduler több rendszert támogat:
Lineáris bemelegítés: Fokozatosan növeli az LR-t nulláról a célig.
Koszinusz-csökkentő: Csökkenti az LR-t egy koszinusz görbét követve a konvergencia biztosítása érdekében.

Hardverintegráció: B200 és Kernel Fusion

Az SFD-optimalizálót a B200MemoryManager és a kernelfúziós stratégiák révén a modern GPU-architektúrákhoz optimalizálták.

B200MemoryManager (TMEM)
Ez az összetevő kezeli a Blackwell osztályú GPU-kon elérhető Tensor Memory (TMEM)-ot. Biztosítja, hogy a sebesség és a Fisher pufferek a gyors chipmemóriában legyenek lokalizálva, hogy minimalizálják a HBM sávszélesség szűk keresztmetszeteit.
Lefoglalás: allocateTMEM(size) blokkokat foglal le a hardveres gyorsítású tenzormemóriakészletben.

Kernel Fusion
Az SFD úgy hajtja végre a „kernel-fúziót”, hogy a lendületfrissítést, a Fisher-átlós frissítést és a paraméteralkalmazást egyetlen GPU-kernel-lépésben egyesíti. Ez csökkenti az optimalizálási lépésenkénti memória-visszautazások (betöltések/tárolások) számát.

Kód entitás leképezés
Rendszerkoncepció - Zig osztály/szerkezet - Fájl hivatkozás
:--- - :--- - :---
Optimizer Core - SFD
Tenzor adatok - Tensor
Memóriakezelő - B200MemoryManager
Szóráscsökkentő - MARS
Előkondicionáló - SophiaSOAP

Elosztott képzés

Az elosztott képzési alrendszer biztosítja az infrastruktúrát a több GPU-t és a hibrid kvantum-klasszikus modellek optimalizálásához. Kihasználja az NCCL-t (NVIDIA Collective Communications Library) a nagy teljesítményű GPU-GPU kommunikációhoz és a Futhark-ot a felgyorsított kernelvégrehajtáshoz. A rendszer támogatja a súly-delta átlagolást a rangok között, az adatkészlet-particionálást és a szinkron akadályprimitíveket.

Építészet és koordinációA GPUCoordinator elsődleges interfészként szolgál az elosztott állapotú és a kollektív műveletek kezeléséhez. Inicializálja az NCCL kommunikátort, kezeli a CUDA adatfolyamokat, és absztrakciót biztosít a betanítási hurok során használt szabványos kollektív műveletekhez.

GPUCoordinator inicializálása
Az inicializálási folyamat több lépésből áll annak biztosítására, hogy az összes rangot szinkronizálják és hozzárendeljék a megfelelő hardverhez:
1. Eszközválasztás: A rangsorok a rank % local_device_count használatával vannak leképezve a helyi GPU-kra.
2. NCCL beállítás: A ncclUniqueId minden rangban meg van osztva (általában megosztott fájlrendszeren keresztül), és a ncclComm inicializálására szolgál.
3. Stream létrehozása: Egy dedikált cudaStream_t jön létre a számítással való átfedő kommunikációhoz.
4. Akadályok lefoglalása: A barrier() megvalósításának megkönnyítése érdekében kis GPU-puffer van lefoglalva.

Kollektív műveletek
A koordinátor becsomagolja az NCCL primitíveket, hogy kezelje az adatmozgást a world_size-en keresztül:
allReduce: Az összes GPU tenzorait összesíti (pl. gradiens átlagoláshoz).
broadcast: Szinkronizálja a súlyokat a gyökér rangtól (Rang 0) az összes többi rangig.
allGather: A részeredményeket minden rangról egyetlen nagy pufferbe gyűjti.
reduceScatter: Csökkenti az adatokat, és szétszórja az eredményt a rangok között.

Akadály megvalósítása
A barrier() funkció biztosítja, hogy minden rang elérjen egy adott végrehajtási pontot a folytatás előtt. Egy allReduce műveletet használ egy barrier_buffer álon, hogy kikényszerítse a szinkronizálást az NCCL kommunikátoron keresztül.

Adatfolyam: Elosztott koordináció
A következő diagram a magas szintű DistributedTrainer és a mögöttes NCCL/CUDA primitívek közötti kapcsolatot szemlélteti.

Elosztott oktatók

A JAIDE két elsődleges tréner-megvalósítást kínál: egy szabványos DistributedTrainer-t a hibrid kvantummunkaterhelésekhez, és egy DistributedTrainerFuthark-t, amely a Futhark kernelekkel történő tiszta GPU-teljesítményre van optimalizálva.

DistributedTrainerFuthark
Ez a tréner a f16 pontosságú és 100%-ban VRAM-rezidens képzésre összpontosít. A RSFAccelerator-t használja a Futhark által generált GPU-kóddal való interfészhez.

Súly-delta átlagolási minta:
A hagyományos SGD helyett, ahol a gradienseket átlagolják, a Futhark edző gyakran súly-delta mintát alkalmaz:
1. A helyi rangok a particionált adatkészletük frissítéseit számítják ki.
2. A allReduce művelet a ncclSum-vel együtt kerül meghívásra a változások összesítéséhez.
3. Az eredményt elosztja a world_size-vel a globális frissítés elkészítéséhez.

Adatkészlet particionálás
Az oktatók a JSONL-adatkészleteket a rangok közötti particionálással kezelik, így biztosítva, hogy minden GPU egyedi adatokat dolgozzon fel.
loadDataset: Megnyitja a JSONL fájlt, és kibontja a használható szövegsorokat.
isUsableDatasetLine: JSON-elemző és MGT tokenizátor segítségével ellenőrzi a sorokat, hogy biztosítsa, hogy tokenizálható tartalmat tartalmaznak.
Particionálás: A rangsorok általában kihagynak sorokat, vagy meghatározott tartományokat töltenek be a rank és world_size alapján, hogy megakadályozzák a redundáns számítást.

Ellenőrzőpont-séma
Az ellenőrzőpontok bináris formátumban vannak sorba rendezve (4-es verzió). A séma a következőket tartalmazza:
Fejléc: Verzió és metaadatok.
Modell méretei: model_dim és vocab_size.
Súlyok: Lapított f32 vagy f16 tömbök, amelyek az RSF csatolórétegeket képviselik.
Optimizáló állapot: Lendületpufferek és globális lépésszámlálások.

Felhőintegráció Modallal

A ModalGPUClient megkönnyíti ezeknek az elosztott képzési feladatoknak a felhő-infrastruktúrában való telepítését (például NVIDIA B200/B300 fürtök). Munkatelepítés
Az ügyfél kommunikál a Modal API-val az erőforrások kiépítéséhez és a JAIDE tároló végrehajtásához:
GPU konfiguráció: Speciális hardvert kér, például „B200”, és beállítja a gpu_count értéket (általában 8).
Request Lifecycle: A std.http.Client segítségével POST-kéréseket küld a /v1/functions/deploy számára a model_path és dataset_path kódot tartalmazó JSON-adattartalommal.
Hitelesítés: Minden biztonságos hozzáférési kérelemhez egy „Vivatartó” tokent csatol.

Funkció - Cél
:--- - :---
deployTrainingJob - Új képzési feladatot küld el a Modal felhőbe
getJobStatus - Lekérdezi az API-t egy futó job aktuális állapotáról
sendRequest - Belső segítő a HTTP-kompatibilitás és a fejlécek kezeléséhez

Megvalósítási részletek

Fixpontos aritmetika
Speciális képzési konfigurációk esetén egy egyéni Fixed32_32 típust használnak a nagy pontosságú frissítések kezelésére anélkül, hogy bizonyos kernelekben 64 bites lebegtetések kellenek.

PRNG
Egy egyéni PRNG (pszeudo-véletlenszám-generátor) valósult meg, hogy biztosítsa a súlyozás reprodukálható inicializálását a különböző rangok között, ha osztoznak egy magon.

Alakzat és tenzor logika
Az elosztott rendszer egy Shape struktúrára támaszkodik, amely kiszámítja a lépéseket és a teljes méretet, biztosítva, hogy az NCCL-n keresztül küldött tenzorok egymás mellett legyenek. A isContiguous ellenőrzés kritikus fontosságú a allReduce műveletek végrehajtása előtt a memóriasérülés megelőzése érdekében.

Cloud Training modálissal

A JAIDE v40 rendszer a Modalt használja, hogy méretezhető, szerver nélküli felhőalapú oktatási infrastruktúrát biztosítson. Ezt a környezetet kifejezetten az NVIDIA B200 GPU-k nagy teljesítményű oktatására optimalizálták, kihasználva a Zig fordítót, a Futhark GPU kernel fordítót és a CUDA eszközkészletet integráló, egyedileg épített Docker képfájlokat. A rendszer támogatja az elosztott képzést a 8×B200 csomópontok között, a HuggingFace automatizált adatkészlet-feldolgozását, valamint a modell-ellenőrző pontok állandó tárolását.

Image Build Pipeline

A felhőkörnyezetet egy többlépcsős képalkotási folyamat határozza meg. A kép alapja a nvidia/cuda:12.8.1-devel-ubuntu24.04 (tanításhoz) vagy nvidia/cuda:12.4.0-devel-ubuntu22.04 (következtetéshez), és rendelkezik a JAIDE hibrid architektúrájához szükséges speciális eszközláncokkal.

Építési szakaszok
1. Rendszerfüggőségek: build-essential, git, xz-utils és libgomp1 telepítése.
2. Zig Toolchain: A Zig 0.13.0 telepítése, amely a JAIDE alapmotor fordításához szükséges.
3. Futhark fordító: A Futhark fordító integrálása (éjszakánként vagy a opam-n keresztül) a .fut kernelek C-könyvtárakká történő átalakításához a GPU-gyorsítás érdekében.
4. AOT fordítás: A kép megpróbálja előre lefordítani a Futhark kerneleket és a Zig binárist a képalkotási fázisban, hogy minimalizálja a tároló indítási késleltetését.

Runtime Build tartalék
Ha az előzetes összeállítás meghiúsul vagy a forráskód módosul, a rendszer tartalmaz egy _runtime_build függvényt (vagy _runtime_build_inference a következtetési szkripthez), amely észleli a hiányzó binárisokat, és a végrehajtás megkezdése előtt újrafordítja azokat a futó tárolóban.

Felhő építési és végrehajtási folyamata
Elosztott képzési konfiguráció

Az oktató szkripteket masszív párhuzamosságra tervezték, kifejezetten a 8×B200 GPU konfigurációt célozva.

Erőforrás specifikáció
A modal_distributed_train.py szkript magas szintű erőforrás-követelményeket határoz meg az RSF-modell memóriaigényének kezelésére:
GPU: B200:8.
CPU: 64,0 mag (legfeljebb 80,0).
Memória: 256 GB (262144 MB).
Efemer lemez: 3 TB ideiglenes képzési műtermékek számára. NCCL és GPU környezet
A hatékony több GPU-s kommunikáció biztosítása érdekében a szkript konfigurálja az NVIDIA Collective Communications Library (NCCL) környezetét. Beállítja a NCCL_DEBUG=INFO-t a hibaelhárításhoz, és kifejezetten leképezi a CUDA_VISIBLE_DEVICES-t az észlelt hardver alapján.

Adatok és modellek tartóssága

A modális kötetek állandó tárolást biztosítanak a különböző felhőalapú futtatások között.

Kötet neve - Mount Path - Cél
:--- - :--- - :---
jaide-training-data - /data - Tárolja a feldolgozott finephrase adatkészletet.
jaide-checkpoints - /checkpoints - Tárolja a közbenső edzési állapotokat és a modellsúlyokat.

Dataset Pipeline
A download_finephrase_to_jsonl funkció kezeli a HuggingFaceFW/finephrase adatkészlet feldolgozását. A következő lépéseket hajtja végre:
1. Betölti az adatkészletet a HuggingFace datasets könyvtáron keresztül.
2. Szöveg kivonatolása prioritási kulcsokkal: text, content, sentence vagy article.
3. Szűrők 20 karakternél hosszabb mintákhoz.
4. Az eredményt egy .jsonl fájlba szerializálja, és véglegesíti a kötetet.

Képzési és következtetési belépési pontok

A rendszer két elsődleges modális belépési pontot biztosít: modal_train.py a modelloptimalizáláshoz és modal_inference.py a modellértékeléshez.

Képzési logika (modal_train.py)
A train funkciót @app.function díszíti, amely meghatározza a 8×B200 GPU-igényt. A jaide bináris parancssori végrehajtását hozza létre a következő paraméterekkel:
--mode train
--dataset /dataset/train.jsonl
--epochs, --batch-size, --lr.

Az edzési eredményeket, beleértve az időtartamot és a kilépési kódokat, a rendszer egy training_history.json fájlba naplózza, amely az állandó köteten belül található.

Következtetési logika (modal_inference.py)
A inference függvény kiszolgáló nélküli végpontot biztosít a betanított modell szövegének előállításához. Újratölti a models_volume-t, hogy biztosítsa a legújabb ellenőrzőpontok láthatóságát, szükség esetén végrehajt egy futásidejű összeállítást, és végrehajtja a bináris fájlt a --mode infer-ben.

Entitástársítás: Szkript a bináris interfészhez
Beállítás és üzembe helyezés

A modal_setup.sh szkript automatizálja a felhőkörnyezet inicializálását.

1. Hitelesítés: Ellenőrzi a modális CLI-t, és szükség esetén lefuttatja a modal token new-t.
2. Kötet létrehozása: A jaide-training-data és jaide-dataset kötetet biztosítja.
3. Végrehajtási parancsok: Szabványos sablonokat biztosít a futóedzésekhez egyéni hiperparaméterekkel, például modal run modal_train.py --epochs 100 --dim 1024.

Hardveres gyorsítási réteg

A hardvergyorsítási réteg biztosítja a nagy teljesítményű végrehajtási háttérprogramokat a JAIDE v40-hez. Különféle számítási szubsztrátokat – a GPGPU kernelektől és a CUDA-optimalizált pufferektől az FPGA/ASIC regiszterátviteli szint (RTL) komponensekig – absztrahálja a neurális feldolgozási és optimalizálási alrendszerek által használt egységes felületté.

A réteg három elsődleges tartományra oszlik:
1. GPGPU kernelek: RSF, SSI és Futharkban írt betanítási műveletek adatpárhuzamos megvalósításai.
2. CUDA Bridge: Alacsony szintű Zig-kötések és memóriakezelés az NVIDIA hardverekhez.
3. RTL-összetevők: Hardverleíró logika az alapvető keresési és döntési feladatok egyéni szilícium- vagy FPGA-telepítéséhez.

Kód-rendszer leképezés: Gyorsító interfészek

A következő diagram azt szemlélteti, hogy a magas szintű Zig absztrakciók hogyan lépnek kapcsolatba a mögöttes hardver-specifikus implementációkkal.

Hardver háttér integrációs térkép 7.1 Futhark GPU kernelek
A Futhark könyvtár tartalmazza a rendszer alapvető matematikai kerneleit, OpenCL-re vagy CUDA-ra fordítva. Kezeli a számításigényes Reversible Scatter Flow (RSF) előre- és hátrameneteket, pillangószórási műveleteket és bijektív csatolási logikát használva.

Főbb képességek:
RSF műveletek: rsf_forward_layer és rsf_backward_layer valósítja meg a csatolási matematikai (skálázás/fordítás) és permutációs logikát.
Optimalizálás: fisher_diagonal_update és spectral_natural_gradient implementálása az SFD optimalizálóhoz.
Retrieval: A topk és score_segments gyorsított sorozatkeresést biztosít az SSI alrendszer számára.

A részletekért lásd a Futhark GPU kernelek című részt.

7.2 CUDA kötések és gyorsító interfész
A Zig-CUDA híd biztosítja a szükséges infrastruktúrát az adatok mozgatásához a CPU és a GPU között minimális többletköltséggel. Rögzített memóriát használ a gyors DMA átvitelhez, és kezeli a Futhark által lefoglalt eszköztömbök életciklusát.

Főbb összetevők:
FutharkContext: Kezeli a GPU-eszköz életciklusát és a parancsszinkronizálást.
PinnedMemory: A cudaHostAlloc becsomagolása oldalzárolt memóriapuffereket biztosít, amelyek elengedhetetlenek a nagy sebességű GPU I/O-hoz.
Tömbburkolók: Az olyan típusok, mint a FutharkArray2DF16 típusbiztos fogantyúkat biztosítanak a GPU memóriájában található többdimenziós tenzorokhoz.

A részletekért lásd: CUDA-kötések és gyorsító interfész.

7.3 Clash RTL komponensek
A nem GGPPU-s hardvereken (FPGA-k vagy ASIC-k) történő telepítéshez a JAIDE Clash-ben (egy funkcionális hardverleíró nyelv) írt RTL-összetevőket biztosít. Ezek az összetevők a memória eldöntésére és a keresési gyorsításra összpontosítanak.

RTL architektúra
Főbb összetevők:
MemoryArbiter: Egy 4 kliens Mealy állapotú gép, amely kezeli az egyidejű memória-hozzáférési kéréseket, és biztosítja a méltányos sávszélesség-elosztást a ServiceCycles segítségével.
filterResp: Logika a memóriaválaszok visszairányításához a kérést kezdeményező konkrét ClientID4-hez.

A részletekért lásd: Clash RTL Components.

Futhark GPU kernelek

A Futhark GPU kernelkönyvtár biztosítja a nagy teljesítményű gyorsítási réteget a JAIDE v40 rendszer számára. Megvalósítja az alapvető matematikai műveleteket a Reversible Scatter Flow (RSF) architektúrához, a Stochastic Fisher Diagonal (SFD) optimalizálóhoz és a Structured Sequence Index (SSI) lekéréséhez. Ezeket a kerneleket a CUDA/OpenCL háttérrendszereken való masszív párhuzamosságra tervezték, numerikus biztonsági mintákkal (NaN/Inf kezelés) és fixpontos hardveres szimulációval.

Alapvető RSF-műveletek

Az RSF architektúra bijektív csatolási rétegekre támaszkodik. A Futhark megvalósítás mind előre, mind hátra haladást biztosít, ahol a visszafelé haladást az áramlás reverzibilis tulajdonságai alapján számítják ki, hogy fenntartsák a O(1) memóriahatékonyságot a mélységhez képest.

RSF Forward and Flow
A rsf_forward belépési pont a bemeneti tenzorok kötegeit dolgozza fel. A bemenetet két részre osztja (x_1, x_2), és skálázható és fordítható transzformációt alkalmaz:
1. Skála: y_1 = x_1 \odot \exp(\text{clip}(\text{weights}_s \cdot x_2 + \text{bias}_s)).
2. Fordítás: y_2 = x_2 + (\text{weights}_t \cdot y_1 + \text{bias}_t).

A rsf_scatter függvény pillangós Haar-hullám stílusú keverést valósít meg, egy inv_sqrt2 állandót (1/√2) használva a variancia fenntartásához a transzformáció során. RSF Backward Pass
A rsf_backward kernel a súlyok (s, t) és a torzítások gradienseit számítja ki. Iterálja a köteget, és kiszámítja a dy1_total-t a y_1 közvetlen gradiensének és a y_2 fordítási függvényén keresztül a visszaszaporított gradiensnek a kombinálásával.

Funkció - Szerep - Fájl hivatkozás
:--- - :--- - :---
rsf_forward - Az előremenő bijektív tengelykapcsoló fő bejegyzése
rsf_backward - Gradiens számítás RSF-rétegekhez
rsf_scatter - Bemeneti méretek pillangós keverése
rsf_flow - Belső logika a skála/fordítás csatoláshoz

Adatfolyam: RSF csatolóréteg
A következő diagram a rsf_flow kernelen belüli adatfolyamot mutatja be, bemutatva a felosztott felek és a súlyok közötti kölcsönhatást.

SFD optimalizáló és természetes színátmenet

A Stochastic Fisher Diagonal (SFD) optimalizáló másodrendű információkat használ fel a konvergencia felgyorsítására. A Futhark kernelek kezelik a Fisher információs mátrix frissítését és a természetes gradiens alkalmazását.

Fisher Diagonal frissítés
A fisher_diagonal_update függvény fenntartja a négyzetes gradiensek futó becslését. Tartalmazza a isnan és isinf biztonsági ellenőrzését a gradiens robbanás megelőzése érdekében.

Természetes színátmenet alkalmazása
A spectral_natural_gradient függvény előfeltételezi a gradienst a Fisher-átló inverzével. damping tényezőt használ (alapértelmezett 1e-8f32), hogy biztosítsa a számszerű stabilitást, amikor a Fisher-becslés közel nulla.

Képzési lépések integrációja
A training_step bejegyzés több műveletet egyesít egyetlen GPU-hívásba:
1. batch_forward: Előrejelzéseket számít ki.
2. batch_compute_loss: Kiszámítja az MSE veszteséget.
3. batch_gradients: Kiszámítja az összes paraméter gradienst.
4. sfd_update_half: Frissíti a súlyokat a lendület és a tanulási sebesség segítségével.

Visszakeresés és SSI-kivonatolás

A Structured Sequence Index (SSI) visszakeresési műveleteit speciális pontozási és rendezési kernelek gyorsítják fel.

Pontozás: A score_segments párhuzamos hash egyezést hajt végre a query_hash és a segment_hashes vektora között, és egyezési bónuszt alkalmaz az alappontszámokhoz.
Top-K Selection: A topk gyök rendezést használ (a diku-dk/sorts-ből importált) a legmagasabb pontszámú indexek megtalálásához. A f32_total_order segítségével biztonságosan kezeli a lebegőpontos összehasonlítást a GPU-n.

Fractal LPU szimuláció

A FractalLPU és FractalTile struktúrák rekurzív hardverarchitektúrát szimulálnak a Non-linear Self-Similar Information Retrieval (NSIR) gráfok feldolgozásához. Ez a szimuláció modellezi a NoC (Network-on-Chip) útválasztást és a magkapuzást.

Fraktál dimenzió és kapuzás
A rendszer egy FractalDimensionConfig-t használ, amely meghatározza a hausdorff_dim-t (alapértelmezett 1.5). Ez szabályozza, hogy a FractalTile objektumok hogyan oszlanak fel gyermekekre.

Terheléselosztás és végrehajtás
Load Balancing: A balanceLoad újraelosztja a pending_ops értéket a ComputeUnit tömbök között, ha azok meghaladják a load_balance_factor értéket.
Fixpontos végrehajtás: A executeFixedPoint hardveres aritmetikát szimulál a bemenetek coherence-vel (16 bites fixpontossá alakítva scale) és biteltolásos osztás végrehajtásával.

Rendszerleképezés: Kód-hardverszimuláció
A következő diagram leképezi a Zig entitásokat a szimulált hardverkomponensekre.

Zig-kötések és kontextuskezelés

A futhark_bindings.zig fájl biztosítja az FFI réteget a Zig futási környezet és a lefordított Futhark C kód között. Kontextuskezelés: A futhark_context_new és futhark_context_config_set_device lehetővé teszi a Zig oldal számára a GPU-környezet inicializálását.
Memory Interop: Az átlátszatlan mutatók, mint a struct_futhark_f16_2d, GPU-rezidens tömböket jelentenek. Olyan funkciók, mint a futhark_new_f16_2d adatok feltöltése, míg a futhark_values_f16_2d eredmények letöltése.
Belépési pontok: Minden Futhark entry funkció futhark_entry_ C függvényként jelenik meg, mint például a futhark_entry_rsf_forward.

CUDA kötések és gyorsító interfész

A CUDA Bindings és Accelerator Interface alacsony szintű hidat biztosít a Zig-alapú RSF neurális mag és a GPU hardver között. Ez az alrendszer elvonatkoztatja a memóriakezelést (rögzített hosztmemória vs. eszközmemória), kezeli a CPU és a GPU pufferei közötti verziószinkronizálást, és biztosítja a RSFAccelerator interfészt a nagy teljesítményű kernelek végrehajtásához.

Rendszerarchitektúra

A gyorsítási réteg három különböző szintre tagolódik: a nyers C idegen függvény interfész (FFI) a CUDA számára, a Futhark által generált kernel-összerendelések és a magas szintű Zig RSFAccelerator, amely a kettő közötti adatáramlást irányítja.

Kód entitás kapcsolat
Ez a diagram leképezi a természetes nyelvi összetevőket a sajátos kód-entitásokra és fájlokra.

"Entitásleképezés: Gyorsítási alrendszer"
CUDA-kötések (cuda_bindings.zig)

A cuda_bindings.zig fájl típusbiztos Zig-burkolót biztosít a CUDA Driver és Runtime API-k körül. Kezeli a hibák fordítását a cudaError_t enumokból Zig hibákra.

Billentyűfunkciók
cudaHostAlloc: Oldalzárolt (rögzített) gazdagép memóriát foglal le, amely elérhető a GPU számára, lehetővé téve a nagy sebességű DMA átvitelt.
cudaMemcpy / cudaMemcpyAsync: Szinkron és aszinkron adatátvitel a gazdagép és az eszköz között.
toError: A CUDA visszatérési kódjait CudaError hibakészletté alakító segédprogram.

Gyorsító interfész (accel_interface.zig)

Ez a modul biztosítja a RSFAccelerator-t és a kapcsolódó memóriaabsztrakciókat. Feltételes fordítást használ a gpu_acceleration build opción keresztül annak meghatározására, hogy a GPU kódútvonalaknak aktívnak kell lenniük.

Rögzített memóriakezelés
A PinnedMemory struktúra biztosítja a RAM-ban tárolt neurális hálózati súlyok rögzítését, megakadályozva, hogy az operációs rendszer lemezre cserélje őket, és lehetővé teszi a GPU számára, hogy közvetlen memóriaelérésen (DMA) keresztül hozzáférjen hozzájuk.

alloc(size): A cudaHostAlloc hívása a cudaHostAllocDefault segítségével.
asSlice(T): A nyers mutatót egy Zig-szeletre veti a szabványos tömbhozzáféréshez.

Futhark integráció
A Futharkot használják a nagy teljesítményű GPU-kernelek előállítására az RSF-műveletekhez. A FutharkContext struktúra kezeli a GPU-eszköz környezetének életciklusát.
init(): Konfigurálja a 0-s eszközt, beállítja a csoportméreteket (256) és a csempeméreteket (32) a kontextus inicializálása előtt.
FutharkArray2DF16: A f16 (félpontos) 2D tömbök burkolója, amelyet modellsúlyozáshoz és aktiváláshoz használnak.

RSFAccelerator és adatfolyam

A RSFAccelerator (az RSF modulban van meghatározva, de a accel_interface-t használja) kezeli a f16 súlypufferek szinkronizálását. Felelős a forwardFromTensor műveletért, amely áthelyezi az adatokat a Tensor CPU primitívből a GPU-ra feldolgozás céljából.

Végrehajtási csővezeték
A következő diagram a szabványos Tensor CPU-tól a RSFAccelerator-n keresztül a GPU-kernelek felé történő adatáramlást szemlélteti."Adatfolyam: CPU tenzortól GPU végrehajtásig"
Verziószinkronizálás
A gyorsító egy verziószámlálót tart fenn a súlypufferek számára. Amikor a súlyokat frissítik a CPU-n (például egy optimalizálási lépés során), a RSFAccelerator észleli a verzióeltérést, és elindít egy cudaMemcpy-t, hogy a GPU f16 puffereit szinkronizálja a következő továbbítás előtt.

Hibakezelés

Az interfész egy átfogó AccelError hibakészletet határoz meg a hardverspecifikus hibák kezelésére:
FutharkSyncFailed: Akkor fordul elő, ha a GPU-környezet nem szinkronizálódik a kernel elindítása után.
CudaHostAllocFailed: Akkor fordul elő, ha az illesztőprogram nem tudja lefoglalni a rögzített memóriát, gyakran a rendszermemória nyomása miatt.
InvalidDimensions: Felemelkedik, ha a bemeneti Tensor alakzat nem egyezik a hozzárendelt FutharkArray-vel.

Clash RTL Components

Ez az oldal a Clash-ben (Haskell-to-RTL) megvalósított hardverszintű regisztrációs átviteli szint (RTL) összetevőket dokumentálja. Ezek az összetevők nagy teljesítményű, szintézisre kész hardvermagokat biztosítanak a memória eldöntéséhez, a szekvencia indexeléséhez és rangsorolásához, amelyek az ASIC vagy FPGA telepítését célozzák meg.

A Clash RTL architektúra áttekintése

A hardverelemek a Clash által biztosított szinkron, típusbiztos megközelítéssel lettek megtervezve. A rendszer Mealy állapotú gépeket használ az összetett vezérlőlogika, például a fa bejárása és a több kliens erőforrás-versenyének kezelésére.

Hardver-kód leképezés

A következő diagram áthidalja a funkcionális hardverkövetelményeket az RTL-forrásban meghatározott Haskell-entitások és adattípusok között.

Hardver entitásleképezés

MemoryArbiter

A MemoryArbiter kezeli a megosztott memória-erőforráshoz való hozzáférést legfeljebb 4 egyidejű kliens számára (NumClients = 4). Egy méltányos hozzáférési szabályzatot valósít meg egy Mealy állapotú gép használatával, amely ciklusosan váltja át a kéréseket.

Állapotgép és választottbírósági logika
A választottbíró két elsődleges állapotban működik:
1. ArbIdle: A döntőbíró megkeresi az első elérhető kérést az ügyfélvektortól a findIndex isJust segítségével.
2. ArbServing: Amint egy kliens hozzáférést kapott, a döntőbíró a ServiceCycles által meghatározott meghatározott időtartamra (alapértelmezett 4) kiszolgáló állapotba lép.

Válasz továbbítása
A memóriából érkező válaszokat a rendszer minden kliensnek továbbítja, de a filterResp függvény szűri őket. Ez biztosítja, hogy az ügyfél csak akkor kapja meg a MemResponse azonosítót, ha a respClient azonosító megegyezik a saját ClientID4 azonosítójával.

Alkatrész - Típus - Leírás
:--- - :--- - :---
Addr32 - Unsigned 32 - 32 bites memóriacím.
Data64 - Unsigned 64 - 64 bites adatszó.
ClientID4 - Unsigned 4 - A 4 ügyfél egyikének azonosítója.

SSISearch Core

A SSISearch komponens a Strukturált Sequence Index (SSI) fa hardvergyorsított bejárását valósítja meg. Úgy tervezték, hogy megoldja a SearchRequest lekérdezéseket a memóriában tárolt TreeNode struktúrák bejárásával.

Keresési logikai folyamat
A keresési folyamat egy 3 állapotú ciklus:
1. Üresjárat: SearchRequest-ra vár.
2. Fetching: Memóriakérés kibocsátása egy adott NodeAddr32-hez.
3. Összehasonlítás: A searchKey és a nodeKey összehasonlítása. Az eredménytől függően vagy leáll (talált/nem található), vagy a leftChild vagy a rightChild helyre lép.

Korlátozások és biztonság
Maximális mélység: A keresést a MaxSearchDepthConfig (64) határolja, hogy megakadályozza a végtelen hurkokat a hibásan kialakított fákban.
Null mutatók: A mag a NodeAddr32 0-t nullAddr-ként ismeri fel, ami levélcsomópont-lezáródást jelez.

Keresési állapotátmenetek

RankerCoreA RankerCore egy csővezetékes pontozási motort biztosít a beolvasott szegmensek rangsorolásához. Egy baseScore-t egy számított positionBias-vel kombinál, így finalScore-t állít elő.

Pontozási képlet
A hardver kölcsönös pozíció torzítást valósít meg, hogy megbüntesse a sorozatban később megjelenő szegmenseket:
Pozíció torzítás: positionBiasScale / (segmentPos + 1).
Végső pontszám: baseScore + positionBias.

Rangkövetés
A RankerState követi a lastQuery hash-t. Ha a következő RankRequest objektumok ugyanazon a lekérdezési hash-en osztoznak, a stateCounter növekszik, gyakorlatilag rangindexet rendelve az adatfolyam minden eredményéhez.

Adatstruktúrák
RankRequest: queryHash, segmentID, segmentPos és baseScore tartalmazza.
RankResult: Kiírja a resultID, finalScore és a számított rank értéket.

Szintéziscélok

Minden összetevő meghatároz egy topEntity-t, amely felfedi a szükséges Clock, Reset és Enable jeleket a szabványos FPGA/ASIC szintézis eszközökhöz (pl. Vivado, Quartus vagy Yosys).

MemoryArbiter Top:
SSISearch Top:
RankerCore Top:

Következtetési szerver és API

A JAIDE v40 Inference Server nagy teljesítményű HTTP/1.1 interfészt biztosít a neurális mag- és relációs gráfrendszerekkel való interakcióhoz. Kezeli a kérés teljes életciklusát – a nyers szöveg feldolgozásától és tokenizálásától a reverzibilis folyamatfeldolgozásig és az NSIR-vezérelt modulációig.

Rendszer áttekintése

A szerver a InferenceServer osztály köré épül, amely a dolgozói szálak készletét kezeli a párhuzamos TCP-kapcsolatok kezelésére. Egyéni RateLimiter-t használ a kérési kvóták érvényesítésére, és RESTful API felületet biztosít az állapotfigyeléshez és a modellkövetkeztetéshez.

Következtetési szerver architektúra
A következő diagram a HTTP-kiszolgáló összetevői és a mögöttes feldolgozómotor közötti kapcsolatot szemlélteti.

"Következtetési kiszolgáló összetevői"

API felület

A kiszolgáló két elsődleges végpontot tesz közzé szabványos HTTP/1.1-en keresztül.

Végpont - Módszer - Leírás
:--- - :--- - :---
/v1/health - GET - Visszaadja a szerver állapotát, az üzemidőt és a modell betöltési állapotát.
/v1/inference - POST - A szöveget az RSF/NSIR folyamaton keresztül dolgozza fel.

Kérelem- és válaszsémák
A kérelmek JSON-objektumként kerülnek elküldésre. A InferenceRequest struktúra határozza meg a várt mezőket, beleértve a text és az opcionális max_tokens bemenetet. A válaszok InferenceResponse objektumokként jelennek meg, amelyek tartalmazzák a generált tokens, opcionális embeddings és nagy pontosságú processing_time_ms kódokat.

Kiszolgáló konfigurálása és végrehajtása

A szerver konfigurálása a ServerConfig struktúrán keresztül történik, amely szabályozza a hálózati paramétereket (port, gazdagép, maximális kapcsolatok), a biztonságot (API-kulcskövetelmények) és a teljesítményt (kötegméret, sebességkorlátok).

A jaide-inference-server végrehajtható fájl elemzi a parancssori argumentumokat, hogy felülbírálja ezeket az alapértelmezett értékeket.

Kód-rendszer leképezés
Ez a diagram áthidalja a CLI konfigurációs logikáját a belső kiszolgáló állapotával.

"CLI a kiszolgáló inicializálásához"

Részletes megvalósítási modulok

A következtetési kiszolgáló felelőssége megoszlik a kérés életciklus-kezelése és az ellenőrzött végrehajtási motor között. InferenceServer – HTTP API és Request Lifecycle
Ez a gyermekoldal részletezi a InferenceServer belső mechanikáját. A következőkre terjed ki:
Request Lifecycle: Manuális HTTP/1.1 fejlécelemzés és InferenceRequest érvényesítés.
Processing Pipeline: A sorozat a MGT tokenizálástól a RSFLayer beágyazásig és a SSI indexelésig.
Utófeldolgozás: Hogyan állítja be a nsirModulateForInference a kimeneteket a relációs gráf állapota alapján.
Memóriastratégia: Kérelemenkénti ArenaAllocator használata a szivárgásmentes, nagy egyidejű teljesítmény biztosítása érdekében.

Verified Inference Engine és ZK Proofs
Ez a gyermekoldal a VerifiedInferenceEngine-t fedi le, amely kriptográfiai garanciákat nyújt a modell kimenetére. A következőkre terjed ki:
ZK Proofs: Integráció a ZKInferenceProver-vel és a Circom következtetési nyomkövetéssel.
Adatvédelem: A Laplace-zaj alkalmazása a differenciált adatvédelem érdekében.
Integrity: Blake3 kötelezettségvállalási sémák és BatchVerifier gördülő kivonatellenőrzés a kérelmek kötegeiben.

InferenceServer – HTTP API és Request Lifecycle

A InferenceServer az elsődleges interfész a külső fogyasztók számára a JAIDE v40 rendszerrel való interakcióhoz. Nagy teljesítményű HTTP/1.1 API-t biztosít, amely levezényli az átmenetet a nyers szövegbevitelről a neurális beágyazásra és az indexelt visszakeresésre. A kiszolgálót nagy párhuzamosságra tervezték, többszálú architektúrával, kérésenkénti aréna memóriakezeléssel és egyedi gördülő ablak sebességkorlátozóval.

Kiszolgáló konfigurálása és inicializálása

A szerver konfigurálása a ServerConfig struktúrán keresztül történik, amely meghatározza a hálózati paramétereket, a biztonsági követelményeket és a modell elérési útvonalait.

Mező - Típus - Alapértelmezett - Leírás
:--- - :--- - :--- - :---
port - u16 - 8080 - Port a TCP figyelő számára.
host - []const u8 - "127.0.0.1" - Kötési cím.
max_connections - u32 - 100 - Maximális egyidejű TCP-kapcsolatok.
rate_limit_per_minute - u32 - 10 - IP-címenként engedélyezett kérések 60-as ablakban.
require_api_key - bool - true - A X-API-Key fejlécek érvényesítése.
max_request_size_bytes - usize - 1MB - Biztonsági korlát a bejövő HTTP törzsekre.

HTTP API végpontok

A szerver manuális HTTP/1.1 elemzőt valósít meg a többletterhelés minimalizálása és a külső függőségek elkerülése érdekében.

1. Szerezze meg a /v1/health-t
Visszaadja a kiszolgáló aktuális állapotát, az üzemidőt, és azt, hogy a modellsúlyok sikeresen betöltődnek-e a memóriába.
Válaszséma: HealthResponse
Mezők: status, uptime_seconds, model_loaded, version.

2. POST /v1/inference
A szövegfeldolgozás elsődleges belépési pontja. Elfogad egy JSON hasznos adatot, és visszaadja a tokenazonosítókat és az opcionális beágyazásokat.
Séma kérése: InferenceRequest
Válaszséma: InferenceResponse

Életciklus és adatfolyam kérése

Egy következtetési kérés életciklusa több szakaszból áll, a hálózati rétegtől a neurális csővezetéken keresztül és vissza.

Pipeline Architecture
A következő diagram a HTTP-kérésről a belső feldolgozó entitásokra való átmenetet szemlélteti.

Következtetési kérelem csővezeték
Megvalósítási részletek Díjkorlátozás
A RateLimiter egy 60 másodperces gördülő ablakot használ, amelyet egy StringHashMap RequestLog struktúrán keresztül valósít meg. Minden napló nyomon követi egy adott IP-címről érkező legutóbbi kérések időbélyegét.
Mechanizmus: Amikor egy kérés érkezik, a checkAndRecord meghívódik. Levágja a 60 másodpercnél régebbi időbélyegeket, és ellenőrzi, hogy a fennmaradó szám meghaladja-e a max_requests értéket.
Parakuurencia: A szál biztonságát a RateLimiter globális mutexe és az egyes RequestLog egyedi mutexek tartják fenn.

Memóriakezelés
A nagy teljesítmény biztosítása és a szivárgások elkerülése érdekében a szerver kérésenkénti ArenaAllocator-t használ.
Minden bejövő kapcsolat létrehoz egy szálat (vagy készletet használ), ahol az arénát inicializálják.
Minden közbenső struktúra – InferenceRequest elemzés, MGT token puffer és InferenceResponse karakterlánc – ezen az arénán belül van lefoglalva.
A HTTP-válasz elküldése után az egész arénát felszabadítják, biztosítva az O(1) tisztítást.

Feldolgozási logika: Tokenizálás modulációvá
1. Tokenizálás: A MGT (morféma-vezérelt tokenizáló) a bemeneti szöveget tokenazonosítók sorozatává alakítja.
2. Beágyazás: A tokenek áthaladnak a RSFLayer-n, hogy nagydimenziós vektorokat generáljanak.
3. SSI indexelés: A vektorok lekérdezése a SSI (Strukturált Sequence Index) alapján történik, hogy megtalálják a releváns történelmi kontextust vagy tudáshorgonyokat.
4. Utófeldolgozás: A nsirModulateForInference függvény (a nsir_core.zig-től) a kvantumrelációs gráf aktuális állapota alapján állítja be a kimeneti beágyazásokat, biztosítva, hogy a válasz kontextuálisan megalapozott legyen az NSIR tudásbázisban.

Hibakezelés

A szerver szabványos HTTP állapotkódokat használ a hibaállapotok kommunikálására:

Kód - Forgatókönyv - Kód Entitás
:--- - :--- - :---
400 Bad Request - Érvénytelen JSON vagy hiányzik a "text" mező. - error.InvalidJson, error.MissingTextField
401 Unauthorized - Hiányzik vagy érvénytelen X-API-Key (ha szükséges). - ServerConfig.require_api_key
429 Too Many Requests - A díjkorlát túllépve. - RateLimiter.checkAndRecord
500 Internal Error - A modell nincs betöltve, vagy az elosztás nem sikerült. - error.OutOfMemory, model_loaded == false

Rendszerentitás-leképezés

Ez a diagram áthidalja a fogalmi "Következtetés" folyamatot az egyes szakaszokért felelős Zig forrásfájlokkal és struktúrákkal.

Kód entitás leképezés

Verified Inference Engine és ZK Proofs

A VerifiedInferenceEngine titkosításilag biztonságos végrehajtási környezetet biztosít a JAIDE v40 neurális következtetéseihez. Integrálja a Zero-Knowledge (ZK) igazolásokat, a differenciális adatvédelmet (DP) és a kötelezettségvállalási sémákat, hogy biztosítsa, hogy a modell kimenetei mind helyesek (matematikailag bizonyítottan az adott modellsúlyokból származnak), mind pedig privátak (védve legyenek az adatszivárgás ellen zajinjektálás révén).

Architektúra és adatfolyam

A motor több részösszetevőt hangszerel a ZKInferenceProof létrehozásához. Kezeli a modellsúlyok, a végrehajtási nyomok és a mögöttes ZK-próba életciklusát.

Rendszer interakciós diagram

Ez a diagram a magas szintű VerifiedInferenceEngine és a zk_verification.zig és dataset_obfuscation.zig kriptográfiai primitívek közötti kapcsolatot szemlélteti.

Alapkomponensek

VerifiedInferenceEngine
A központi koordinátor a biztonságos következtetésért. Inicializálható ZK támogatással vagy anélkül a init vagy initWithZKProofs használatával. Súlykezelés: Betölti és tárolja a s (skála) és t (fordítás) függvények rétegsúlyait az RSF architektúrában.
Verification Tracking: Fenntartja a verification_count és successful_verifications paramétereket a motor integritásának figyeléséhez.
Modell integritás: model_hash-t (Blake3) használ, amely állandó magból származik, hogy biztosítsa, hogy a modell architektúráját ne manipulálják.

ZKInferenceProver
A híd a SNARK (Groth16) hátteréhez. Kezeli a lebegőpontos következtetési műveletek áramkör-kompatibilis fixpontos tanúkká alakítását.

Circom Integration: .circom áramköröket fordít, és a CircomProver segítségével kezeli a kulcsokat.
Witness Generation: A f32 tenzorokat fixpontos egész számokká alakítja a precision_bits (alapértelmezett 64) használatával az áramköri fogyasztáshoz.
Kötegelt ellenőrzés: BatchVerifier-t használ gördülő hash-sel a több bizonyítás hatékony érvényesítéséhez.

Differenciális adatvédelem (DP)
A tagsági következtetések elleni támadások megelőzése érdekében a motor Laplace-zajt fecskendez a következtetési eredményekbe.
Laplace Noise: A DifferentialPrivacy.applyLaplaceNoise-ben implementálva.
Adatvédelmi költségvetés: epsilon, delta és sensitivity paraméterekkel kezelhető.

ZK áramkör: inference_trace.circom

Az ellenőrzött következtetés alapvető logikáját a Circom határozza meg, amely a Groth16 bizonyító rendszert célozza meg. Fixpontos aritmetikát végez az RSF (Reversible Scatter Flow) számítás szimulálásához.

Logikai áramlás az áramkörben
Az áramkör ellenőrzi, hogy egy adott x bemenetnél a y kimenetet helyesen számították ki a lekötött modellsúlyok használatával.

Kulcsáramkör-sablonok
Sablon - Cél - Fájl
:--- - :--- - :---
RSFLayerComputation - Megvalósítja a kapcsolóréteget: y = x \cdot \exp(s) + t.
PoseidonChain - Hatékonyan kivonatolja a nagy bemeneti vektorokat egyetlen mezőelembe.
VerifyMerkleProof - Érvényesíti, hogy a súlyok a modell lekötött súlyfájához tartoznak.
RangeProof - Biztosítja, hogy a beinjektált DP zaj a megengedett adatvédelmi határokon belül maradjon.

Megvalósítási részletek

Fixpontos RSF számítás
Mivel a ZK áramkörök véges mezőkön működnek, a motor a f32 értékeket egész számokká alakítja a FIXED_POINT_SCALE (1 000 000) segítségével. Az RSF csatolóréteg exponenciális függvényét az áramkörön belüli 3. rendű Taylor-kiterjesztés segítségével közelítjük meg:
Lineáris együttható: 1
Kvadratikus együttható: 0,5
Köbegyüttható: 0,166667

Bizonyítási összesítés
A ProofAggregator Merkle-fát használ, hogy több következtetési bizonyítást egyetlen gyökérben egyesítsen. Ez lehetővé teszi a rendszer számára, hogy ellenőrizze a N bonyolultságú N következtetések kötegét.
Merkle Tree: A ProofAggregator-n keresztül valósul meg.
Ellenőrzés: A verifyBatch függvény ellenőrzi a teljes fa integritását.

Elkötelezettség és ujjlenyomat
A motor biztosítja az adatok elkülönítését a DatasetFingerprint és CommitmentScheme használatával.
Blake3 Commitments: Gyors helyi kötelezettségvállalásokhoz használják az adatok beviteléhez.
Adatkészlet ujjlenyomata: Egyedi, 32 bájtos azonosítót hoz létre a betanító készlet számára, hogy biztosítsa, hogy a modellt ne finomhangolják jogosulatlan adatokon.

Biztonság, biztonság és formális ellenőrzésA JAIDE v40 kódbázis többrétegű, mélyreható védelmi stratégiát valósít meg a neurális-relációs műveletek helyességének és a feldolgozott adatok biztonságának biztosítása érdekében. Ez az alrendszer az alacsony szintű futásidejű memória őröktől a biztonsági modellek, például a Bell-LaPadula és a Biba magas szintű matematikai bizonyítékaiig terjed. Azáltal, hogy a formális ellenőrzést közvetlenül a core_relational folyamatba integrálja, a JAIDE biztosítja, hogy az információáramlás konzisztens maradjon a meghatározott biztonsági szabályzatokkal még összetett kvantumrelációs érvelés során is.

Rendszer áttekintése

A biztonsági architektúra három elsődleges tartományra oszlik:
1. Helyesség és ellenőrzés: Hoare logika és formális invariánsok felhasználása a SelfSimilarRelationalGraph integritásának bizonyítására.
2. Információfolyam-vezérlés: Rács alapú biztonsági szintek és integritási szintek érvényesítése az összes rendszerelvben és objektumban.
3. Adatvédelem: Homomorf titkosítás és adathalmazok elhomályosítása az érzékeny információk védelmére a betanítás és a következtetések során.

Biztonsági és ellenőrzési topológia

A következő diagram az ellenőrző motor, a biztonsági szabályzat végrehajtója és az alapvető adatstruktúrák közötti kapcsolatot szemlélteti.

Biztonsági logika a kód entitásleképezéshez

Hivatalos ellenőrzési és biztonsági igazolások

Az ellenőrző alrendszer biztosítja a JAIDE megbízhatóságának matematikai alapot. Strukturált bizonyítási rendszert használ az olyan invariánsok karbantartására, mint a CONNECTIVITY, SYMMETRY és MEMORY_SAFETY.

Főbb összetevők:
Biztonsági modellek: A Bell-LaPadula modell (nincs leolvasás, nincs leírás) és a Biba integritási modell (nincs leolvasás, nincs írás) megvalósítása.
Rács alapú hozzáférés-vezérlés: PUBLIC és TOP_SECRET közötti biztonsági szintek és UNTRUSTED és KERNEL integritási szintek.
Formális igazolások: FormalVerifier, amely ProofRule készleteket (pl. MODUS_PONENS, INDUCTION, FRAME_RULE) alkalmaz a gráf állapotátmeneteinek ellenőrzésére.

A részleteket lásd: Formális ellenőrzés és biztonsági igazolások.

Biztonság, zavarás és C API

A biztonsági réteg futásidejű védelmet nyújt a gyakori szoftversérülékenységek ellen, és titkosítási elfedéssel biztosítja az adatok védelmét.

Főbb összetevők:
Runtime Safety: A safety.zig modul ellenőrzött castingot (safeIntCast, safePtrCast) és biztonságos memóriaműveleteket biztosít, mint például a secureZeroBytes, hogy megakadályozza az érzékeny kulcsok kiszivárgását.
Dataset Obfuscation: A Paillier homomorf titkosítást valósítja meg, lehetővé téve matematikai műveletek (összeadás, skaláris szorzás) végrehajtását közvetlenül a titkosított rejtjelezett szövegeken, dekódolás nélkül.
Foreign Function Interface: A c_api.zig-ben meghatározott C-kompatibilis API felület, amely lehetővé teszi a külső alkalmazások számára, hogy biztonságosan kommunikáljanak a JAIDE maggal, miközben fenntartják a Zig-ben meghatározott biztonsági határokat.

Adatvédelem és biztonsági folyamat
Részletekért lásd: Biztonság, eltüntetés és C API.

Biztonsági konfigurációs állandók

A rendszer a biztonsági paraméterek központosított készletét használja az obfuszkáció erősségének és a hozzáférési jogok részletességének meghatározására.

Állandó - Érték / bit - Leírás
:--- - :--- - :---
SECURITY_PARAMETER - 256 - Bithossz a kriptográfiai biztonság érdekében
ACCESS_RIGHT_READ_BIT - 1 - Bitmaszk olvasási hozzáféréshez
ACCESS_RIGHT_ADMIN_BIT - 16 - Bitmaszk adminisztrátori jogosultságokhoz
PRIME_P - u256 - Előre meghatározott nagy prime a Paillier generációhoz

Hivatalos ellenőrzési és biztonsági igazolásokA JAIDE v40 rendszer szigorú, többrétegű biztonsági és ellenőrzési architektúrát valósít meg. Ez a rendszer biztosítja, hogy az információáramlás megfeleljen a formális biztonsági modelleknek, a gráfinvariánsok megmaradjanak a kvantumrelációs műveletek során, és a végrehajtási nyomok kriptográfiailag auditálhatók legyenek. A megvalósítás fel van osztva a formal_verification modulra a logikai helyesség, a security_proofs a hozzáférés-vezérlés és az információáramlás, valamint a z_runtime a biztonságos végrehajtás érdekében.

Biztonsági modellek és hozzáférés-vezérlés

A security_proofs.zig modul a Bell-LaPadula (Bizalmasság) és Biba (Integrity) biztonsági modelleket valósítja meg. Ezeket a modelleket a biztonsági és integritási szintek rácsalapú összehasonlításával kényszerítik ki.

Biztonsági és integritási szintek
A rendszer különálló szinteket határoz meg a titoktartás és az integritás tekintetében:
Biztonsági szint: PUBLIC (0) – TOP_SECRET (4).
IntegrityLevel: UNTRUSTED (0) – KERNEL (3).

Formális biztonsági szabályok
A rendszer az alábbi szabályokat érvényesíti az illegális információáramlás megelőzése érdekében:
1. Egyszerű biztonsági tulajdonság (nincs felolvasás): Egy adott SecurityLevel-nél lévő alany nem tud olvasni egy magasabb SecurityLevel-vel rendelkező objektumot.
2. Csillagtulajdonság (nem írható le): Egy alany nem írhat olyan objektumra, amelynek értéke alacsonyabb SecurityLevel.
3. Biba integritás (nincs leolvasás/nem írható fel): Az alanyok nem tudnak alacsonyabb integritási szintekről olvasni, vagy magasabb integritási szintekre írni.

Biztonsági modell adatfolyam
A következő diagram bemutatja, hogy a SecurityLevel és a IntegrityLevel hogyan működnek együtt a SecurityProofsConfig-on belül a hozzáférés érvényesítése érdekében.

Biztonsági érvényesítési logika
Formális ellenőrzés és változatlanok

A formal_verification.zig modul keretet ad a SelfSimilarRelationalGraph helyességének bizonyítására. InvariantType definíciókat használ az NSIR gráf állapotának és matematikai konzisztenciájának figyelésére.

Invariáns típusok és prioritások
A rendszer előnyben részesíti a biztonság szempontjából kritikus invariánsokat a szerkezetiekkel szemben:
MEMORY_SAFETY (10-es prioritás): Biztosítja, hogy ne legyen puffertúlcsordulás vagy érvénytelen mutatóhivatkozás.
TYPE_SAFETY (9. prioritás): Érvényesíti a kvantum- és relációs típuskonverziókat.
KAPCSOLAT ÉS KOHERENCIA: Biztosítja a gráf topológiát és a kvantumfázis konzisztenciáját.

Bizonyítási szabályok
A modul egy logikai motort valósít meg a ProofRule alapján. Ezek a szabályok lehetővé teszik a rendszer számára, hogy biztonsági tulajdonságokat származtasson axiómákból:
MODUS_PONENS: 2 helyiség szükséges.
TEMPORAL_INDUCTION: Az időbeli állapotátmenetek ellenőrzésére szolgál.
LOOP_INVARIANT: Biztosítja a gráf bejárási stabilitását.

Z-Runtime Safety Layer

A z_runtime.zig modul a relációs műveletek felügyelt végrehajtási környezeteként működik. Karbantart egy teljes ExecutionHistory-t, hogy auditálható nyomvonalat biztosítson a rendszeren belüli minden átalakításról.

Végrehajtás ellenőrzése
A ZRuntime által végrehajtott minden művelet ExecutionHistoryEntry-ként kerül rögzítésre. Ez a következőket tartalmazza:
Akciótípusok: create_variable, relational_operation, entangle_variables, measure.
Metaadatok: Időbélyegek (nanoszekundumos pontossággal), elsődleges/másodlagos célok és eredményértékek.

Változó életciklus és tulajdonjog
A ZVariable struktúra egy SelfSimilarRelationalGraph-t és a hozzá tartozó RelationalQuantumLogic-t foglal magába. Nyomon követi saját history és creation_order kódját, hogy biztosítsa, hogy az állapotváltozások ellenőrizhetők legyenek a biztonsági modulokban meghatározott formális szabályok szerint. Futásidejű entitásleképezés
Ez a diagram leképezi a „Változó végrehajtás” természetes nyelvi fogalmait a belső Zig entitásokra.

Z-futásidejű végrehajtási folyamat
Kriptográfiai integráció

Az ellenőrző réteg nagy teljesítményű kriptográfiai primitíveket használ a bizonyítékok és biztonsági címkék integritásának biztosítására.

Alkatrész - Primitív - Cél
:--- - :--- - :---
CommitmentScheme - Blake3 - Gyors, biztonságos állami kötelezettségvállalás a ZK igazolásokhoz.
Biztonsági igazolások - Sha256 / Sha512 - Biztonsági leírók és integritáscímkék kivonatolása.
Ellenőrzés - timingSafeEql - Állandó idejű összehasonlítás az oldalcsatornás támadások megelőzésére.

Kivonatmegvalósítás: A std.crypto.hash.sha2.Sha256 és Sha512 kódot használja egyedi azonosítók generálására biztonsági kontextusokhoz.
Homomorf integráció: Míg a szabványos tenzorok feldolgozása normálisan történik, az ellenőrző réteg támogatja a Paillier homomorf titkosítással való integrációt a modellsúlyok adatvédelmi megőrzése érdekében (lásd: SecurityError.CryptographicError).

Biztonság, zavarás és C API

Ez a szakasz a futásidejű integritásért, adatvédelemért és együttműködési képességért felelős alrendszereket tárgyalja. A safety modul robusztus futásidejű védelmet biztosít a gyakori memória- és aritmetikai hibák megelőzésére. A dataset_obfuscation modul magánélet-megőrző technikákat valósít meg, beleértve a Paillier homomorf titkosítást is, az érzékeny képzési adatok kezelésére. Végül a c_api.zig egy idegen funkciós interfészt (FFI) biztosít a JAIDE NSIR és optimalizálási képességeinek külső C/C++ környezetekbe való integrálásához.

Biztonsági és futásidejű őrök

A biztonsági modul egy átfogó segédprogram-csomagot valósít meg, amely a rendszer integritásának érvényre juttatását szolgálja futás közben. Ezeket a segédprogramokat a kódbázisban használják a kézi memóriakezeléssel és az alacsony szintű bitmanipulációval kapcsolatos kockázatok mérséklésére.

Alapvető biztonsági funkciók
Safe Casting: safeIntCast és safeUsizeToInt határellenőrzést hajt végre a IntegerOverflow és IntegerUnderflow elkerülése érdekében.
Mutató érvényesítése: A safePtrCast biztosítja, hogy a mutatók ne legyenek nullák, a céltípushoz megfelelően igazodjanak, és érvényes eredetük legyen.
Memória nullázása: A secureZeroBytes és secureZeroSlice illékony írásokat használ annak biztosítására, hogy az érzékeny adatok törlődnek a memóriából, és ne a fordító optimalizálja azokat.
Állandó idejű összehasonlítás: A secureCompare időzítési támadásnak ellenálló bájt-összehasonlítást biztosít.

Secure Utilities
A modul SecureRng burkot is tartalmaz a std.crypto.random és MonotonicClock körül a nagy pontosságú, elcsúszásálló időméréshez.

Adatkészlet obfuszkáció és homomorf titkosítás

A JAIDE a dataset_obfuscation modult használja az érzékeny adatkészletek kezelésére anélkül, hogy a nyers értékeket bizonyos feldolgozási szakaszokban felfedné. Ez elsősorban a Paillier kriptorendszer egyedi megvalósításával érhető el.

Paillier kriptorendszer megvalósítása
A PaillierKeyPair tárolja az aszimmetrikus homomorf titkosításhoz szükséges nyilvános és privát összetevőket (n, g, \lambda, \mu). Titkosítás: A encrypt a i64 egyszerű szöveget u512 titkosított szöveggé alakítja. 256 bites előjel-bites kódolást használ a negatív egész számok támogatására.
Dekódolás: A decrypt visszaállítja az eredeti egész számot a privát \lambda és \mu paraméterek használatával.
Homomorf műveletek:
add: Két rejtjelezett szöveget megszoroz, hogy az összegükből titkosított szöveget állítson elő.
multiplyScalar: A rejtjelezett szöveget skaláris hatványra emeli, így a termék titkosított szövegét állítja elő.

Elfojtott adatfolyam
A következő diagram azt szemlélteti, hogy az adatok hogyan alakulnak át egyszerű szövegből feldolgozás céljából obfuszkált állapotba.

Adat obfuscation Pipeline

C API és FFI Bridge

A c_api.zig fájl határozza meg a JAIDE külső interfészét, lehetővé téve a C-kompatibilis nyelvek interakcióját a nsir_core és a EntangledStochasticSymmetryOptimizer nyelvekkel.

Fogantyú alapú architektúra
Az API átlátszatlan mutatókat használ a belső Zig állapot biztonságos kezelésére:
CGraph: Átlátszatlan fogantyú a GraphContext-hez, amely beburkolja a SelfSimilarRelationalGraph-et, és mutex a menetbiztonság érdekében.
COptimizer: A EntangledStochasticSymmetryOptimizer átlátszatlan fogantyúja.

Optimalizálás és statisztika
A EntangledStochasticSymmetryOptimizer szimulált lágyítási megközelítést valósít meg a gráf energiaminimalizálására. A folyamatot a OptimizationStatistics segítségével követi, amely rögzíti az iterációkat, az elfogadási arányokat és az energiaszinteket.

C API integrációs leképezés
A következő diagram áthidalja a C-szintű függvényhívásokat a belső Zig implementációkhoz.

C API a belső logikai leképezéshez

Hibakódok
Az API c_int állapotkódokat ad vissza, jelezve a sikert vagy az adott hibamódot:
Állandó - Érték - Leírás
:--- - :--- - :---
JAIDE_SUCCESS - 0 - A művelet sikeresen befejeződött
JAIDE_ERROR_NULL_POINTER - -1 - Egy null mutatót adtak át az API-nak
JAIDE_ERROR_ALLOCATION - -2 - A memóriafoglalás nem sikerült
JAIDE_ERROR_NODE_NOT_FOUND - -3 - A célcsomópont nem létezik a grafikonban
JAIDE_ERROR_OPTIMIZATION_FAILED - -6 - Az optimalizálónak nem sikerült konvergálnia

Szószedet

Ez a szószedet határozza meg a JAIDE v40 kódbázisban használt műszaki terminológiát, tartomány-specifikus fogalmakat és építészeti primitíveket. A JAIDE egy nagy nyelvi modell, amely a Reversible Scatter Flow (RSF) paradigmán alapul, és a CPU-k, GPU-k és kvantumrelációs gráfok közötti nagy teljesítményű végrehajtásra tervezték.

Alapvető építészeti feltételek

RSF (Reversible Scatter Flow)
A JAIDE alapvető neurális architektúrája. A transzformátorokkal és a CNN-ekkel ellentétben az RSF bijektív csatolási rétegekre épül, amelyek biztosítják, hogy minden előremenő műveletnek pontos algebrai inverze legyen. Ez lehetővé teszi a O(1) memória bonyolultságát a visszaterjesztés során, mivel az aktiválások menet közben rekonstruálhatók a kimenetekből.
Megvalósítás: LayerCore struct in.
Billentyűfunkciók: forwardInPlace és inverseInPlace.

NSIR (nemlineáris önhasonló információkeresés)
Hierarchikus érveléshez használt kvantumrelációs gráfrendszer. A tudást csomópontok (kvantumállapotokkal) és élek (összefonódási és fraktál tulajdonságokkal rendelkező) gráfjaként ábrázolja.
Megvalósítás: SelfSimilarRelationalGraph hüvelyk.
Fogalmak: A qubitek, az összefonódás és az összetett amplitúdók a relációs megbízhatóság ábrázolására szolgálnak. OFTB (Ortogonális fraktál transzformációs blokk)
Paraméter nélküli keverőblokk, amely Haar-wavelet butterfly-transzformációt használ az információ elosztására a tenzordimenziók között. Felváltja a Transformersben található önfigyelési mechanizmust.
Megvalósítás: rsf_scatter kernel be.
Logika: inv_sqrt2 (1/√2) skálázási tényezőt használ a variancia fenntartásához.

SFD (Sztochasztikus Fisher-átló)
Másodrendű optimalizáló, amelyet RSF-modellek betanításához használnak. Becslései szerint a Fisher Information Matrix átlója természetes színátmenet-frissítéseket hajt végre, vegyes precíziós támogatással kombinálva (FP4-FP64).
Megvalósítás: SFDOptimizer hüvelyk.
Alkatrészek: Tartalmazza a MixedPrecisionTrainer és B200MemoryManager.

Adatstruktúrák és primitívek

Term - Meghatározás - Fájl hivatkozás
:--- - :--- - :---
Tensor - A többdimenziós tömbök elsődleges adatszerkezete, amely támogatja a másolást írásban (COW) és a referenciaszámlálást.
MGT - Morféma-vezérelt tokenizer. Háromszintű csővezeték: speciális tokenek → morfológiai dekompozíció → BPE tartalék.
SSI - Strukturált szekvencia index. A szekvenciaszegmensek Hamming-távolságú hasonlósági kereséséhez használt 64 gyűjtőhelyes hash-fa.
LayerCore - A JAIDE „számítási egysége”, amely 4 tanítható tenzort tartalmaz: s_weight, t_weight, s_bias, t_bias.
Qubit - Egy csomópont állapotának reprezentációja az NSIR-ben, amelyet két komplex amplitúdó (alfa és béta) határoz meg.

Logikai diagramok feldolgozása

Természetes nyelvből kódolt entitásleképezés: Következtetési folyamat
A következő diagram leképezi a magas szintű következtetési fogalmakat a kódbázis adott osztályaihoz és függvényeihez.

Tudásgráf Evolúció: CREV Pipeline
A CREV (Collective Relational Evolution and Validation) folyamat áthidalja a szövegbevitelt az NSIR kvantumgráfjával.

Memória és hardver fogalmak

Aréna / Födém / Buddy allokátorok
A JAIDE egyéni allokátorok hierarchiáját használja a töredezettség minimalizálása és az átviteli sebesség maximalizálása érdekében a magas egyidejűségű következtetések során.
Aréna: Gyors, lineáris elosztás a kérésenkénti életciklusokhoz.
Buddy: Az NSIR-gráf dinamikus, két hatvány méretű blokkjaihoz használatos.

Futhark kernelek
GPU-gyorsított háttérrendszer. A Futhark kód OpenCL-re vagy CUDA-ra van fordítva, és megbirkózik a rsf_flow és natural_gradient frissítésekkel.
Butterfly Mixing: A Haar-szerű szórási művelet megvalósítása.
Fisher frissítés: Az SFD optimalizáló négyzetes gradienseinek mozgóátlaga.

GPUCoordinator
Kezeli a több GPU-s kommunikációt az NCCL (NVIDIA Collective Communications Library) segítségével. Kezeli a szinkronizálási akadályokat és az olyan kollektív műveleteket, mint a allReduce a súly-delta átlagolásához.
Megvalósítás: GPUCoordinator hüvelyk.
Gomb funkció: ncclCommInitRank az elosztott csoportok inicializálásához.

Matematikai szimbólumok a kódban

s\_weight (W_s): A exp(W_s \cdot x_2 + b_s) affin skálázási tényező kiszámításához használt skálasúlymátrix.
t\_weight (W_t): Az affin eltoláshoz használt fordítási súlymátrix W_t \cdot y_1 + b_t.
inv\_sqrt2 (1/√2): Az OFTB-ben használt fraktálskálázási állandó az energia megőrzésére a pillangó transzformáció során.
Komplex amplitúdók (a, b): Az NSIR gráfban annak valószínűségét jelzik, hogy egy csomópont - 0\rangle vagy - 1\rangle állapotban legyen.
