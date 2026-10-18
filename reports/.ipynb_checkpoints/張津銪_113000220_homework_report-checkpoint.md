# Homework Report: Reddit Stock Sentiment (3 classes)

**Name:** 張津銪 · **Student ID:** 113000220

I ran the same 18 tools from AgentDev, without changing any code, on `newdataset/Reddit-stock-sentiment.csv`: 847 Reddit posts and comments labelled `-1.0` (negative), `0.0` (neutral) or `1.0` (positive). Every number below comes from a tool result in my session log and is cited by its `result_id`. Statements the tools could not confirm are marked as hypotheses. Throughout, I compare with my 20-newsgroups run (AgenticPipeline) to check whether the pipeline generalises.

## 1. Data and quality checks

`load_dataset_1` loaded **847 documents**: `0.0` 423, `-1.0` 315 and `1.0` 109. The positive class is only about 13% of the data, and the labels arrive as the strings `'-1.0'`, `'0.0'` and `'1.0'`. All my tools handled three string labels without modification, so on this point the pipeline generalises. The first five rows (`inspect_data_2`) are very short (16, 137, 33, 1,079 and 24 characters), and some labels look questionable for a stock forum. *"Seeing lots of red in the ticker"* is labelled neutral, although "red" means falling prices. *"Calls on retards"* is labelled negative, although "calls" is a bullish bet. Row 3 reads like a 1,079-character press release but is labelled positive. Row 284 (`inspect_data_3`) is an auto-generated post: `&#x200B;`, an empty markdown link to Coinbase and a list of crypto tickers (`$BTCUSD $ETHUSD $JASMYUSD $PEPEUSD`). Here the noise is markdown/HTML leftovers and bot posts, not the e-mail headers of 20-newsgroups.

`check_duplicates_5` (on the `text` column) found **23 extra copies involving 34 rows**. `inspect_data_6` showed that rows 34 and 42 are the same human-written comment labelled `1.0`. The other duplicated rows I inspected (rows 284 and 160) are bot templates labelled `0.0`. Because `drop=True` in my tool uses `keep=False`, dropping would have removed all 34 rows, including both copies of a genuine positive comment in an already small class. I therefore **kept the duplicates** and tracked their effect on later steps instead. `describe_data_7` shows right-skewed lengths in every class (mean / median / max in characters): `-1.0` 153.7 / 96 / 2,040, `0.0` 140.8 / **68** / 2,753, `1.0` 164.6 / 82 / 2,461. Typical posts are 10–20 times shorter than the newsgroup posts (medians 883–1,649), and the neutral class has the lowest median because it contains one-word replies such as *"Cash"*.

![Text length by category](../plots/張津銪_homework_plot_describe_data_7.png)

## 2. Document-term matrix and term frequency

The default DTM had a vocabulary of **4,346 terms** (`term_freq_9`), and its top terms were function words (*the* 920, *to* 566, *and* 488). With English stop words removed, `build_dtm_10` has shape **(847, 4105)** (241 columns removed), 9,906 non-zero entries and 99.72% sparsity. At about 3.5 million cells it is far below the 20-million-cell limit of my reduction tool, so, unlike in the newsgroups run, no `max_features` cap was needed. The stop-word top 30 (`term_freq_11`) is driven by the April 2025 tariff news rather than by format: *market* 111, *just* 95, *trump* 92, *like* 83, *china* 72, *tariffs* 39. Of the template tokens I was looking for, only *x200b* (the `&#x200B;` zero-width space) made the list, at rank 23 with 28 counts. *stockbuyvest* and *com* were not in the top 30. My hypothesis that bot templates would dominate raw frequency was therefore only partly right. *don* (34) is again a tokenisation leftover of "don't".

## 3. Variance filter

With the same threshold as on newsgroups (0.1), `variance_12` kept **only 9 of 4,105 terms**: *market* 0.428, *futures* 0.181, *trump* 0.161, *stock* 0.160, *star* 0.143, *like* 0.133, *china* 0.127, *just* 0.118, *price* 0.107. On newsgroups the same threshold kept 807 of 5,000. Short texts make most counts 0 or 1, and a 0/1 column has variance of at most 0.25. *market* (0.428) exceeds that bound, so the survivors are exactly the terms that are **repeated inside a few documents**, not the broadly used ones. I traced two of them to specific rows:

- *star* comes from row 169 (`inspect_data_13`), a 2,461-character promotional post about Star Navigation Systems labelled `1.0`.
- *futures* comes from the bot template *"how does the futures market look today"* (`inspect_data_14`, row 160, 759 characters). In its first 500 characters *futures* and *market* each appear 5 times, and the template exists in 4 identical copies (rows 160, 352, 473, 836).

This is the same mechanism as in the newsgroups run, where a few long graphics FAQ posts dominated the variance filter. Lowering the threshold to 0.02 (`variance_15`) kept 118 terms. Several terms just below the cut have exactly the same variance (*monitoring*, *navigation*, *services*, *sna*, *speed*, *syntheia*: 0.018868), almost equal to *chinese* (0.0198). My explanation is that a word repeated several times in one post and a word used once each by many people can produce the same variance. Variance alone cannot tell "one post repeating a word" from "many people using it".

## 4. Correlation filters and term correlations

**Neutral class (`pearson_16`, |r| ≥ 0.1, 10 kept).** This step confirmed my template hypothesis. Every positively correlated term is a bot-template token: *com* and *stockbuyvest* (0.142), *updated* (0.125), *x200b* (0.106), *weekly* (0.104) and *lower* (0.100). *weekly* fits the "Updated Weekly" template, and *lower* plausibly comes from the futures template ("a lower S&P futures price"). The negatively correlated terms are the tariff topic: *economy* (−0.152), *trump* (−0.129), *trade* (−0.118). In other words, "neutral" is partly defined by bot posts, and "not neutral" by politics. The strongest |r| is only 0.152, versus 0.380 for sci.med in the newsgroups run, so the class signal is weak.

![Pearson correlation with the neutral class](../plots/張津銪_homework_plot_turn114.png)

**Positive class (`pearson_17`, 41 kept).** The top terms mix promotional language and casual praise: *huge*, *main* and *tldr* (all 0.155133), *nice* 0.154, *technology* 0.151, *potential* 0.131, and *changing*, *god*, *major*, *perfectly* (all 0.127811). The repeated identical values have a simple cause. For a binary term column, Pearson's r equals the phi coefficient, which depends only on how many documents contain the term (n) and how many of them are positive (k). With N = 847 and 109 positive documents, a term in a single positive post scores only φ = 0.0895, and one in two positive posts scores 0.1266. So |r| ≥ 0.1 effectively requires a term to appear in at least two documents, mostly positive ones. The removed terms just below the cut (all 0.095784) include *downtrend* and *dumping*. Bearish words that lean positive are another hint of label noise.

**Negative class (`spearman_18`, 18 kept).** *trump* (0.161), *fuck* (0.144), *economy* (0.119), *trade* (0.115), *china* (0.112), *americans* and *trust* (0.111) and *retard* (0.110) correlate with negative posts, while *com* and *stockbuyvest* are negative (−0.123). The negative class is the only one whose top terms describe real content: anger about the tariff conflict. On these sparse counts Spearman behaves almost like a presence/absence measure. The roughly 830 zeros of a term share one low average rank, so the jump from 0 to 1 dwarfs the difference between 1 and 3.

**Term-term correlations (`feature_corr_19`, top 20 by variance).** The strongest links are artefacts, not word associations. *star*–*aviation* = 1.00 because both occur only in row 169. *market*, *futures*, *stock* and *price* form a block with r = 0.71–0.83 (for example *futures*–*stock* 0.827), created by the four copies of the futures template. Ordinary words (*trump*, *china*, *think*, *money*) correlate weakly with each other.

![Top-20 term correlation matrix](../plots/張津銪_homework_plot_feature_corr_19.png)

## 5. Frequent pattern mining

- **Neutral, term-frequency filter, FP-Growth, min_sup = 5 (`patterns_20`).** 2,431 terms were kept and 592 patterns found. The maximum support was only 6, because the 95th-percentile cut removed the frequent words, the same problem as `patterns_15` on newsgroups. The only multi-term patterns in the top 20 are the crypto tickers: {btcusd, ethusd, pepeusd} with support 6, at least 5 of which come from the duplicated Coinbase template (row 284 and its copies). This is duplication directly creating a "frequent" pattern.
- **Neutral, TF-IDF filter, min_sup = 10 (`patterns_21`).** I raised min_sup because TF-IDF keeps frequent terms and the templates could make itemsets explode. 2,023 terms were kept and 47 patterns found. The top single items are *market* 37, *like* 36, *just* 31, *trump* 23, *china* 22, and the only top-20 multi-term pattern is {stockbuyvest, com} with support 20, the bot's link.
- **Negative, TF-IDF filter, min_sup = 5 (`patterns_22`, 173 patterns).** The top 20 were all single items (*trump* 45, *just* 42, *china* 33, *tariffs* 16). An itemset can never have higher support than its own single items (the Apriori property), so pairs cannot outrank them.
- **Negative, MaxFP-Growth (`patterns_23`, 146 maximal itemsets).** This run surfaced the pairs: {trump, china}, {trump, just} and {trump, people}, each with support 9.

These are the only topical co-occurrences in the whole run, but they cover under 3% of the 315 negative posts. Short texts give pattern mining very little to work with.

## 6. Dimensionality reduction and document similarity

PCA (`reduce_dims_24`) explains only 6.37% + 5.52% = 11.89% of the variance, against 23.56% on newsgroups. The plot shows a single `1.0` point far out at about (25.7, −14.4) and a single `0.0` point at about (7, 11.3), while almost all other posts sit near the origin. Based on the variance results, I believe the first is row 169. The second is the futures template, and `cosine_sim_25` supports this: rows 160 and 352 have cosine similarity 1.0, so the four copies are identical vectors stacked on one point. As on newsgroups, the two PCA axes mainly describe outlier documents. UMAP (`reduce_dims_26`, n_neighbors = 15) shows the three sentiment classes as **one fully mixed cloud**, unlike newsgroups, where christian and atheism separated. Two small tight groups appear: an orange-only (`0.0`) stack near (−1, 3.9) and a mixed-colour group near (2.5, −2.6). I suspect these are duplicated templates or posts left empty after stop-word removal, and ticker/`x200b`-style posts, but the tool output cannot map points to rows, so these remain hypotheses.

![PCA projection](../plots/張津銪_homework_plot_reduce_dims_24.png)

![UMAP projection](../plots/張津銪_homework_plot_reduce_dims_26.png)

`binarize_27` produced a (847, 3) one-hot matrix in sorted order (`-1.0`, `0.0`, `1.0`) with counts 315 / 423 / 109, matching the loading step. Finally, the two promotional posts that drove the positive-class signal share almost no vocabulary (`cosine_sim_28`, rows 3 and 169: 0.05998). One is about electric boats, the other about aviation safety. The positive class is therefore not held together by a shared vocabulary.

## 7. Conclusion and limitations

The tools generalised without code changes: three string labels, shorter texts and a smaller matrix all worked, and the only error was an out-of-range parameter (`max_chars` > 500), which my validation caught as intended. The analysis itself shows that **bag-of-words on these short posts captures format, duplication and topic, not sentiment**. The neutral class is partly defined by bot templates (`pearson_16`, `patterns_20`/`21`). The positive class is shaped by a few promotional posts (`variance_12`, `pearson_17`, `cosine_sim_28`). PCA is driven by two outlier sources (`reduce_dims_24`, `cosine_sim_25`), and UMAP shows no sentiment separation (`reduce_dims_26`). Only the negative class carries a clear topic (*trump*, *china*, *tariffs*). This mirrors the newsgroups finding, where Usenet headers and signatures outweighed topic, but the source of noise is different here.

In a second iteration I would:

1. Strip HTML/markdown entities (`&#x200B;`, empty links) before tokenising.
2. Remove bot templates by de-duplicating with `keep="first"` or filtering the stockbuyvest source.
3. Use binary or TF-IDF weighting so that one long post cannot dominate variance and PCA.
4. Add bigrams so that negations and slang ("not good", "buy calls") keep their meaning.

The labels themselves look noisy (rows 0 and 2 in `inspect_data_2`), which caps what any word-based method can achieve. The tool summaries only show the top 10–20 items, so some links (which rows form the UMAP groups, how many of the 6 ticker documents are copies) remain unverified.
