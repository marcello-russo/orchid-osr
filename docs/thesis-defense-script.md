# Copione Discussione Tesi — OrchID OSR

> Risposte pronte per ogni critica prevedibile. Struttura:
> **Critica** → **Riconoscimento** → **Difesa** → **Lavoro futuro**

---

## 1. "Perché detection rate e non AUROC?"

**Riconoscimento**: "È una scelta discussibile, lo riconosciamo. L'AUROC è lo standard in OSR."

**Difesa**: "Abbiamo scelto detection rate al 10% FPR perché più interpretabile per un pubblico non specialistico — '273 immagini su 300 riconosciute' è più concreto di 'AUROC 0.954'. Però in tutte le tabelle riportiamo anche l'AUROC come metrica secondaria, e i risultati sono consistenti: ConvNeXt-Tiny ha AUROC 0.954."

**Chiusura**: "Per la pubblicazione scientifica useremo AUROC come primaria, come da standard."

---

## 2. "Una specie novel sola non basta"

**Riconoscimento**: "È la limitazione principale dello studio."

**Difesa**: "Abbiamo usato O. sphegodes_Napoli perché era l'unica popolazione del complesso con ≥50 immagini certamente non appartenenti alle classi note. È una limitazione esplicitamente discussa nella tesi."

**Chiusura**: "Per il futuro, abbiamo già identificato O. apifera e O. insectifera come candidate — appartengono a cladi diversi e permetterebbero una valutazione multi-specie."

---

## 3. "Dove sono le barre d'errore?"

**Riconoscimento**: "Giusta osservazione. Le abbiamo aggiunte."

**Difesa**: "Nella versione finale della tesi, tutte le tabelle di cross-validation riportano media ± deviazione standard tra i 5 fold. Per esempio, DINOv2-Base in Exp 7 ha CV 0,933 ± 0,006 — la variabilità è molto bassa."

**Chiusura**: "Abbiamo anche calcolato il silhouette score come metrica oggettiva di separabilità degli embedding."

---

## 4. "Come sapete che il miglioramento DINOv2 non è solo full FT?"

**Riconoscimento**: "È un problema reale. DINOv2-Base passa da 0,687 (Exp 7, LoRA) a 0,911 (Exp 8, full FT)."

**Difesa**: "Due elementi mitigano questa critica. Primo: DINOv2-Small migliora solo di 0,016 (0,888 → 0,904), suggerendo che per modelli bilanciati il full FT non è il fattore dominante. Secondo: il risultato principale — ConvNeXt-Tiny detection rate 0,907 — NON è affetto da questo problema, perché ConvNeXt era in full FT in TUTTI gli esperimenti."

**Chiusura**: "Resta una limitazione, che riconosciamo. Per separare i due effetti servirebbe un esperimento con DINOv2 full FT in SupCon, che non abbiamo fatto per costi computazionali."

---

## 5. "Perché non avete testato MGCE puro?"

**Riconoscimento**: "Non l'abbiamo fatto. Sarebbe stato istruttivo."

**Difesa**: "Exp 10 nasce come tentativo esplorativo: integrare MGCE in SpHOR per vedere se la multi-granularità potesse migliorare l'OSR. Il collasso a Palena è stato così netto e qualitativo (74% delle predizioni su una singola classe) che abbiamo ritenuto il risultato sufficiente per concludere che l'integrazione non funziona, indipendentemente dal contributo del singolo componente."

**Chiusura**: "Concordo che un test MGCE standalone sarebbe più pulito. È nei lavori futuri."

---

## 6. "L'ablation study dov'è?"

**Riconoscimento**: "Non abbiamo fatto ablation sistematica."

**Difesa**: "Ogni esperimento in full FT su V100S costa ~8 ore. Abbiamo preferito allocare le risorse su 6 architetture × 3 paradigmi = 18 esperimenti completi piuttosto che su ablation per un singolo modello. Iperparametri (τ=0.07, λ=0.01) ripresi dalla letteratura originale."

**Chiusura**: "Un'analisi di sensitività su τ e λ è il primo candidato per lavori futuri."

---

## 7. "t-SNE non è affidabile"

**Riconoscimento**: "t-SNE ha limitazioni note — Wattenberg et al. (2016) le documentano."

**Difesa**: "Per questo affianchiamo al t-SNE una metrica oggettiva: il silhouette score con distanza coseno. I risultati confermano le osservazioni qualitative: DINOv2-Base ha silhouette 0,076 (SupCon), che indica una separabilità modesta ma misurabile."

**Chiusura**: "Per la pubblicazione, valuteremo l'uso di UMAP come alternativa più stabile."

---

## 8. "Il dataset è piccolo, 50 immagini di test per classe sono poche"

**Riconoscimento**: "50 immagini per classe di test è un limite statistico — una singola immagine vale 2 punti di F1."

**Difesa**: "È il più grande dataset per Ophrys ultra-FG esistente, raccolto in anni di campagne. La CV 5-fold con stratificazione e il test set bilanciato garantiscono che le metriche siano robuste entro i limiti del dato disponibile."

**Chiusura**: "L'espansione del dataset è prioritaria. Stiamo valutando collaborazioni con iNaturalist e citizen science."

---

## 9. "Non ci sono immagini degli errori"

**Riconoscimento**: "Mancano, è vero."

**Difesa**: "Abbiamo analizzato quantitativamente i pattern di errore dalle matrici di confusione. La coppia exaltata↔sphegodes è costantemente la più frequente (12-20%), seguita da garganica↔incubacea — un pattern coerente con la tassonomia del complesso."

**Chiusura**: "Una figura con esempi di errori sarebbe un ottimo contributo per la camera di commercio — lo aggiungeremo prima della pubblicazione."

---

## 10. "Il risultato negativo è troppo forte"

**Riconoscimento**: "Testiamo 2 metodi su 1 dataset — non possiamo generalizzare a tutto l'ultra-FGVC."

**Difesa**: "La conclusione nella tesi è formulata con cautela: 'con i metodi testati e su questo specifico dataset'. Non affermiamo che l'NCD in ultra-FGVC è impossibile, ma che rimane un problema aperto."

**Chiusura**: "Riformuleremo per enfatizzare ancora di più la specificità del risultato."

---

## 11. "Il data leakage non è veramente risolto"

**Riconoscimento**: "MD5 rimuove solo copie identiche, non near-duplicate."

**Difesa**: "Le immagini provengono da fonti diverse (campo 2019-2025, internet da archivi botanici) scattate in anni e contesti diversi. La probabilità che due foto near-duplicate finiscano in split diversi è bassa. Inoltre, il test set è composto esclusivamente da immagini non presenti in training — verifica manuale."

**Chiusura**: "Un'analisi di similarità coseno tra split sarebbe un controllo aggiuntivo utile."

---

## 12. "Domain shift: foto da campo vs internet"

**Riconoscimento**: "Le specie da campo (majellensis, Palena) hanno accuratezza perfetta in tutti gli esperimenti — è un potenziale confound."

**Difesa**: "Due elementi dimostrano che il modello apprende anche caratteri morfologici genuini. Primo: la confusione simmetrica O. exaltata → O. sphegodes (12–20%) coinvolge due specie ENTRAMBE da internet. Se il modello discriminasse solo per dominio fotografico, tutte le specie da internet sarebbero indistinguibili tra loro. Secondo: O. garganica ↔ O. incubacea è un altro pattern di confusione reciproca tra specie da internet — seguono la tassonomia, non la fonte."

**Chiusura**: "Resta una limitazione, ma non invalida i risultati. Il pattern di confusione è biologicamente plausibile e non spiegabile dal solo dominio fotografico."

## 13. "Il silhouette score dei vostri embedding è vicino a zero"

**Riconoscimento**: "I silhouette score per SupCon e SpHOR sono effettivamente bassi (0.04–0.08 per DINOv2, negativi per ConvNeXt e ResNet)."

**Difesa**: "È atteso e coerente con il dominio ultra-FG. La silhouette misura separabilità in spazio coseno. Se le classi sono intrinsecamente simili (differenze minime tra specie dello stesso complesso), la separabilità sarà modesta indipendentemente dal metodo. Infatti anche CE, che è il miglior classificatore closed-set, ha silhouette solo 0.66 per ResNet-18. Non è un problema del metodo — è una proprietà del dominio."

## 14. "Avete aggiunto error analysis: non dimostra che il modello sbaglia tanto?"

**Riconoscimento**: "Le confusioni più frequenti sono 12–20% — non sono trascurabili."

**Difesa**: "Ma sono BIOLOGICAMENTE SIGNIFICATIVE. exaltata↔sphegodes e garganica↔incubacea sono le coppie tassonomicamente più vicine del dataset. Il fatto che il modello le confonda con la stessa frequenza con cui si confondono i tassonomi umani (Devey et al. 2008 riportano 15–30% di disaccordo) suggerisce che il modello ha imparato pattern morfologici reali, non artefatti."

---

## Frase finale per ogni critica

> "È un limite che riconosciamo, lo abbiamo discusso nella tesi, e abbiamo già
> pianificato di affrontarlo nei lavori futuri."

Se usi questa, il professore non può insistere. Hai dimostrato consapevolezza,
priorità e piano. La seduta è un esame di consapevolezza, non di perfezione.
