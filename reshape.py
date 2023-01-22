# import libraries
import pandas as pd

# import original data with their original labels

davidson = pd.read_csv("https://raw.githubusercontent.com/t-davidson/"
                       "hate-speech-and-offensive-language/master/data/labeled_data.csv")
trac = pd.read_csv("https://raw.githubusercontent.com/kmi-linguistics/trac-1/master/english/agr_en_train.csv", header=None)

jig = pd.read_csv("https://raw.githubusercontent.com/katkorre/tox-reannotation/data/toxkaggle.csv")

# rename some columns and drop some to facilitate restructuring and merging with new annots
davidson = davidson.rename({'tweet':'text'}, axis=1)
davidson = davidson.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
trac = trac.rename({1:'text'}, axis=1)
trac =trac.drop(0, axis=1)
jig = jig.rename({'comment_text':'text'}, axis=1)
jig = jig.rename({'toxkaggle':'original_label'}, axis=1)

# read new annotation files
hits1 = pd.read_csv("https://raw.githubusercontent.com/katkorre/tox-reannotation/data/offensive1.csv"")
hits2 = pd.read_csv("https://raw.githubusercontent.com/katkorre/tox-reannotation/data/offensive2.csv")
hits3 = pd.read_csv("https://raw.githubusercontent.com/katkorre/tox-reannotation/data/offensice3.csv")


# reformat data
def restructure(hits):
    group = hits.groupby("_unit_id", axis="rows")
    data = pd.DataFrame()
    data["text"] = group.text.apply(lambda x: list(x)[0])
    data["gold_toxicity"] = group.toxicity_gold.apply(lambda x: list(x)[0])
    data["toxicity"] = group.toxicity.apply(lambda codes: [1 if c == "YES" else 0 for c in codes])
    data["source"] = group.dataset_code.apply(lambda x: list(x)[0])
    # data = data.iloc[:120,:]
    data = data[pd.isnull(data['gold_toxicity'])]
    data = data.drop(['gold_toxicity'], axis=1)
    data[['annotator_1_label', 'annotator_2_label', 'annotator_3_label', 'annotator_4_label',
          'annotator_5_label']] = pd.DataFrame(data.toxicity.tolist(), index=data.index)
    data = data.drop(['toxicity'], axis=1)
    data['source'] = data['source'].replace({'A': 'DavidsonHS', 'B': 'DavidsonOFF', 'C': 'Trac1', 'D': 'Toxkaggle'})
    return data

# make separate dataframes for each round in order to join them later
data1 = restructure(hits1)
data2 = restructure(hits2)
data3 = restructure(hits3)

# merge them
data_new = data1.merge(data2, how='outer', on=['text', 'source'])
data = data_new.merge(data3, how='outer', on=['text', 'source'])

# add original annotations
# need to reconstruct the dataframe a bit as to be in the multilabel format.
# create binary columns for trac one
trac.loc[trac[2] == 'OAG', 'OAG'] = 1
trac.loc[trac[2] == 'NAG', 'NAG'] = 1
trac.loc[trac[2] == 'CAG', 'CAG'] = 1

# drop initial column
trac = trac.drop([2], axis=1)

# do the same with Davidson
davidson.loc[davidson['class'] == 0, 'hate_speech'] = 1
davidson.loc[davidson['class'] == 1, 'offensive'] = 1
davidson = davidson.drop(['class'], axis=1)

data_d = data.merge(davidson, how='left', on='text')
data_dj = data.merge(jig, how='left', on='text')
df = data.merge(trac, how='left', on='text')

# write final file to csv
df.to_csv('offensive.csv', index=False)
