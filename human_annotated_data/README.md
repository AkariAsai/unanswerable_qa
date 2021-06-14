## Human Annotated data
This directory includes the human annotated data. 

### File names

The table shows the annotated file names for each dataset / language. You can also see the annotated data in a spreadsheet format [here](https://drive.google.com/file/d/1rAroUr5gvU__jSjCMWErYN9-i0R_dsxM/view?usp=sharing).


| Dataset      | language | # of annotated questions | file name | 
| ----------- | ----------- |----------- |----------- |
| Natural Questions    | English | 450 | [NQ.tsv](NQ.tsv) |
| TyDi QA    | Bengali | 50 | [TyDi-Bn.tsv](TyDi-Bn.tsv) |
| TyDi QA      | Japanese | 100 | [TyDi-Ja.tsv](TyDi-Ja.tsv) |
| TyDi QA     | Korean | 100 | [TyDi-Bn.tsv](TyDi-Ko.tsv) |
| TyDi QA     | Russian | 50 | [TyDi-Ru.tsv](TyDi-Ru.tsv) |
| TyDi QA     | Telugu | 50 | [TyDi-Te.tsv](TyDi-Te.tsv) |


### Data Format
The annotation is in a tab separated file. The first row is a header. 
- `data`: This column represents if the data is from TyDi QA or Natural Questions (i.e., `tydi` for the TyDi QA data and `nq` for the Natural Questions data. 
- `language`: This column presents the languages of the questions. 
- `question`: This column presents the original questions. 
- `Translated question`: This is only for the TyDi QA data. Authors of this paper recruited professional translators to translate the original TyDi QA questions into English so that we can get better sense of the questions' unanswerabilities. 
- `category`: This represents type of unanswerability, as defined in the paper. (1) **factoid question**, (2) **invalid answers**, (3) **false premise**, (4) **non-factoid question**, (5) **invalid questions**, (6) **multi-evidence question**.
- `answer`: This column records a new annotated answer (if exists).
- `original_page`: This column shows the originally annotated web page where the example is from.
- `answer_page`: This column represents the webpage where the new annotated answer is found.
- `answer in different_wiki`: This column shows if the `answer` is from another Wikipedia page or not.
- `answer in non-Wikipedia article on the web`: This column shows if the `answer` is found in the web outside Wikipedia.
- `answer in infobox/table`: This column represents whether the new answer is found in the infobox of Wikipedia page.
