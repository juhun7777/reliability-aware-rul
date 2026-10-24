# Definitions for revision questions

These separate code definitions, manuscript statements and historical evidence.
Plausible explanations must not be turned into fabricated experimental history.

| Question | Reproducible definition / action |
| --- | --- |
| alpha, beta, R² | The stated model is R=alpha*F^(-1/3)+beta. The manuscript describes initial0--20000cycle regression. The published code uses constants; the fitting dataset, gains and units need their original calibration record. The evidence script can refit declared data without forcing old coefficients. |
| 12x noise robustness | Define the statistic and compare raw R and recomputed PHI from exactly the same perturbed R/F. The prior figure-generation script used additional raw-only drift/spikes, so its comparison is not equal-input evidence. The new script reports correlation drops, including nonpositive/undefined ratios. |
| rho=0.68 | Specify Pearson(sigma_i,abs(prediction_i-target_i)), observation unit and same checkpoint/MC policy. Do not combine a nine-observation table with a separate many-window array. Whole-cycle rounding sensitivity can be bounded. |
| STFT95% | The original source shuffled overlapping windows80:20 and fit its scaler before the split. The revised MLP separates file/parent groups first. Its accuracy belongs to the new split, not the old95% claim. |
| TableVII Statistics | STFT+MLP uses128 spectral+32 temporal statistics. RandomForest statistics have another extractor. The RUL Bi-LSTM uses PHI,R,F,relativeR,cycle, not that32-stat vector. Define or correct the table's input term before re-running its ablation. |
| 0.7A_2 78.6% | Given EOL345100 and remaining73900, observation271200 yields78.5859%. This inversion assumes the filename matches the table row. It is not proof why the row was selected. Simple floor rounding cannot turn an80--90%cut into78.6%. Export exact IDs/rows/cycles in future runs. |
| TableI and refs13--15 | Verify author, title, venue, DOI and the method actually used, not just a similar title. The listed table citations16--19 map to different works in the bibliography. Correct numbering and study-to-method mapping separately. |

Verified primary records useful for bibliography repair (not automatic
replacements for an unsupported SVM/RandomForest claim):

- H.-C. Teng et al., *Study of Contact Degradation in Final Testing for BGA Socket*, EMAP2007, DOI[10.1109/EMAP.2007.4510290](https://doi.org/10.1109/EMAP.2007.4510290). [Co-author university record](https://paper.isu.edu.tw/paper/class_paper_main.php?dep_id=80202&gType=2&lang_type=&type=1&user_id=18102).
- Meng et al., *Investigation on Intermittent Failure Phenomenon of Electrical Connectors Under Repetitive Mechanical Insertion and Withdrawal Operations*, IEEE TCPMT12(9),1533--1541,2022. [Author university record](https://homepage.hit.edu.cn/renwb).
- M. Ahmad and S. K. Sitaraman, *Study of Mechanical Behavior of Compliant Micro-Springs for Next-Generation Probing Applications*, Journal of Electronic Packaging124,411--418,2002. [Author laboratory record](https://caspar.gatech.edu/journal-publications/).
- B. F. Toler, R. A. Coutu Jr., J. W. McBride, *A review of micro-contact physics for microelectromechanical systems (MEMS) metal contact switches*, JMM23,103001,2013. DOI[10.1088/0960-1317/23/10/103001](https://doi.org/10.1088/0960-1317/23/10/103001). [Author institutional record](https://scholar.afit.edu/facpub/4093/).
- H. Qiu, H. Wang, F. Ke, *Instability of Contact Resistance in MEMS and NEMS DC Switches under Low Force: the Role of Alien Films on the Contact Surface*, Sensors13,16360--16371,2013. DOI[10.3390/s131216360](https://www.mdpi.com/1424-8220/13/12/16360).

The last two are different real papers. Do not combine Toler's authors with
Qiu's title or an unrelated IEEE journal/volume/pages. Exact unmatched records
are marked for correction; search failure alone does not prove nonexistence.
