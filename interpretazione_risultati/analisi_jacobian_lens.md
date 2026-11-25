# Analisi della Jacobian Lens (J-Lens): Confronto con il Paper e Valutazione dei Risultati

**Riferimento principale:** Wes Gurnee, Nicholas Sofroniew, Jack Lindsey, Adam Pearce et al. (2026), *"Verbalizable Representations Form a Global Workspace in Language Models"*, Transformer Circuits Thread ([link](https://transformer-circuits.pub/2026/workspace/#methods-jlens))  
**Implementazione nel progetto:** `src/ws4_lens/jacobian_lens.py`, `docs/CONTRACTS.md` (§9.2), `checkpoints/jacobian_lens/{gpt2,geppetto}.pt`, `results/lens/*.parquet`

---

## 1. Corrispondenza tra implementazione del progetto e paper

### Aspetti conformi al paper

| Aspetto | Paper (Gurnee et al., 2026) | Implementazione del progetto | Valutazione |
|:--------|:----------------------------|:-----------------------------|:------------|
| **Target della derivata** | $\partial \mathbf{h}_{\text{final}} / \partial \mathbf{h}_\ell$ (output dell'ultimo blocco prima della norm finale) | $\partial r_L(t) / \partial h_l(t)$, con `r_L` catturato via forward hook su `model.transformer.h[-1]` **prima** di `ln_f` | **Corretto** (`hidden_states[L]` di HF include già `ln_f`, quindi l'hook è necessario) |
| **Formula di readout** | $W_U\,\text{norm}(J_{\ell \to t}\,\mathbf{h}_\ell)$ | `lm_head(ln_f(h_l @ J_l.T))` (e `lm_head(h_L)` con $J_L = I_d$ all'ultimo layer) | **Corretto** (`h @ J.T` equivale a `(J h^T)^T` per broadcasting su `[B, T, d]`) |
| **Approccio training-free** | Linearizzazione al primo ordine basata solo sui gradienti del modello | Nessun parametro addestrato; media dei gradienti su split di calibrazione (WikiText train per GPT-2, ISDT train per GePpeTto) | **Corretto** |
| **Precisione e ottimizzazione causale** | — | Controllo obbligatorio `float32` (`_check_float32_model`) e troncamento causale del prefisso `0..t_idx` | **Corretto** (bit-for-bit equivalente in modelli causali, evita instabilità `fp16`) |

### Differenze di design rispetto al paper

1. **Approssimazione *same-position* vs *cross-position*:**
   - **Progetto (`CONTRACTS.md §9.2`):** Calcola esclusivamente il Jacobiano alla stessa posizione $J_l = \mathbb{E}_t[\partial r_L(t) / \partial h_l(t)]$, mantenendo fisse le altre posizioni.
   - **Paper:** $J_{\ell \to t}$ è mediato su posizioni sorgente, posizioni target e prompt, includendo anche i percorsi *cross-position* mediati dall'attenzione. La versione del progetto è quindi un'**approssimazione diagonale nel tempo**, ragionevole per la predizione del token successivo a $t$, ma incompleta per l'analisi causale multi-token.
2. **Scala del modello e scopo d'uso:**
   - Nel paper la J-Lens è applicata a modelli molto grandi (famiglia Claude) principalmente per isolare una *J-space* di concetti verbalizzabili ed eseguire interventi causali (*steering*, *patching*, *ablations*).
   - Nel progetto è invece valutata come **lente di decoding token-by-token** su modelli da 124M parametri (`gpt2` e `geppetto`, 12 layer, `d = 768`).

---

## 2. I risultati ottenuti sono sensati?

**Sì.** I risultati osservati in `results/lens/` mostrano un comportamento netto e coerente con la matematica di una linearizzazione statica non centrata.

### Confronto quantitativo (`results/lens/*.parquet`)

| Dataset | Layer | Logit KL | Jacobian KL | Tuned KL | Logit Top-1 | Jacobian Top-1 | Tuned Top-1 |
|:--------|:-----:|---------:|------------:|---------:|------------:|---------------:|------------:|
| `gpt2__pud_en` | **0** | 45.508 | **11.125** | 5.267 | 0.000 | 0.006 | 0.020 |
| `gpt2__pud_en` | **4** | 4.787 | **3.707** | 1.785 | 0.056 | 0.041 | 0.124 |
| `gpt2__pud_en` | **9** | **2.787** | 6.871 | 0.684 | **0.141** | 0.064 | 0.199 |
| `gpt2__pud_en` | **10** | **2.323** | 6.899 | 0.469 | **0.172** | 0.095 | 0.220 |
| `geppetto__pud_it` | **0** | 59.033 | **5.606** | 2.125 | 0.000 | 0.003 | 0.093 |
| `geppetto__pud_it` | **10** | **0.500** | 2.234 | 0.171 | **0.205** | 0.162 | 0.208 |

*(I valori completi di `gpt2__wikitext.parquet` mostrano lo stesso andamento: a L10 Jacobian KL = `8.449`, acc = `0.122` contro Logit KL = `3.095`, acc = `0.240`.)*

### Due fenomeni distinti nei dati

1. **Miglioramento nei layer iniziali (`L0–L6` GPT-2, `L0–L1` GePpeTto):**
   La Jacobian Lens abbatte fortemente la KL rispetto alla Logit Lens all'inizio della rete (es. a `L0` da `45.51` a `11.13` nats su GPT-2, e da `59.03` a `5.61` nats su GePpeTto). Il Jacobiano medio riallinea parzialmente gli embedding di ingresso verso la base finale.
2. **Degrado nei layer medio-alti (`L7–L11`) e mancato anticipo del commitment:**
   Nei layer profondi, la Jacobian Lens **peggiora** rispetto alla Logit Lens sia in KL sia in Top-1 accuracy, e non anticipa il *commitment layer* (90% del massimo):
   - **GPT-2:** UPOS commitment a Layer 12 ($\tau = 1.000$), Token Top-1 a Layer 12 ($\tau = 1.000$).
   - **GePpeTto:** UPOS commitment a Layer 10 ($\tau = 0.833$), Token Top-1 a Layer 12 ($\tau = 1.000$).

---

## 3. Diagnosi matematica dai checkpoint `.pt`

Ispezionando i valori singolari dei checkpoint salvati in `checkpoints/jacobian_lens/`:

- **`gpt2.pt` (`[13, 768, 768]`):**
  - A `L00`: norma di Frobenius `46.48`, `s_max = 18.49`, `s_min = 0.0001`.
  - Ai layer `L07–L11`: la media diagonale cresce regolarmente (`0.630 → 0.965`), ma il valore singolare massimo rimane molto alto (`s_max = 9.87` a `L09`, `8.89` a `L10`, `6.99` a `L11`) con `s_min = 0.0036` (numero di condizionamento $\approx 2740$).
- **`geppetto.pt` (`[13, 768, 768]`):**
  - A `L00`: norma `107.86`, `s_max = 29.73`.
  - Ai layer `L07–L11`: `s_max` è molto più contenuto (`3.50 – 3.66`, `2.98` a `L11`), il che spiega perché su GePpeTto il degrado rispetto alla Logit Lens sia meno marcato che su GPT-2.

### Perché accade

1. **Assenza di centramento e termine di bias:** `h_l @ J_l.T` applica una matrice lineare senza traslazione (`b_l`) né centramento rispetto alle medie di calibrazione ($\bar{r}_L - J_l \bar{h}_l$). Nei layer alti, dove $h_l$ è già vicino alla base di `W_U`, il prodotto per una matrice media non regolarizzata introduce distorsioni che la Tuned Lens evita grazie ai traduttori affini $(A_l, b_l)$ addestrati in KL.
2. **Amplificazione di direzioni ad alto guadagno prima di `ln_f`:** In presenza di valori singolari elevati (`s_max ≈ 9.9` su GPT-2), `J_l` amplifica specifiche direzioni fuori-diagonale; poiché `ln_f` normalizza per la deviazione standard, queste direzioni dominano il vettore normalizzato, rendendo il modello *overconfident* sui token sbagliati (a `L10` su GPT-2 PUD EN l'entropia della J-Lens scende a `2.165` nats con appena `9.5%` di accuracy).
3. **Limite intrinseco riconosciuto in letteratura:** Gli stessi autori di Anthropic, nell'aggiornamento successivo (**J++ lens**, ottobre 2026), hanno evidenziato i limiti di un unico Jacobiano medio globale sul corpus, sostituendolo con Jacobiani condizionati per cluster di attivazioni.

---

## 4. Conclusioni per il report e la presentazione

- **L'implementazione è valida e fedele a `CONTRACTS.md §9.2`**; la restrizione *same-position* è una semplificazione legittima e documentata rispetto a Gurnee et al. (2026).
- **I risultati sono scientificamente sensati e difendibili:** dimostrano empiricamente sia il beneficio locale sui primissimi layer (riduzione della KL rispetto alla Logit Lens) sia i limiti di una linearizzazione statica *training-free* nei layer intermedi/profondi rispetto all'adattamento affine della Tuned Lens.
