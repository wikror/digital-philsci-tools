import pandas as pd
import numpy as np
import pickle
import itertools

DATA_PATH="/home/wikror/gdrive/corpus-study/data/"
# DATA_PATH="/Users/fellow38/Library/CloudStorage/GoogleDrive-w.rorot@uw.edu.pl/My Drive/ornak/corpus-study/data/"
FOS_LIST=["bio", "psych", "bioPsych"]

fields = {
"animalBehavior":
[
"scimagojr 2023  Subject Category - Animal Science and Zoology.csv",
"scimagojr 2023  Subject Category - Insect Science.csv",
"scimagojr 2023  Subject Category - Small Animals.csv",
],

"neuroscience":
[
"scimagojr 2023  Subject Category - Behavioral Neuroscience.csv",
"scimagojr 2023  Subject Category - Cellular and Molecular Neuroscience.csv",
"scimagojr 2023  Subject Category - Cognitive Neuroscience.csv",
"scimagojr 2023  Subject Category - Developmental Neuroscience.csv",
"scimagojr 2023  Subject Category - Neuroscience (miscellaneous).csv",
"scimagojr 2023  Subject Category - Sensory Systems.csv",
],

"psychology":
[
"scimagojr 2023  Subject Category - Developmental and Educational Psychology.csv",
"scimagojr 2023  Subject Category - Experimental and Cognitive Psychology.csv",
"scimagojr 2023  Subject Category - Neuropsychology and Physiological Psychology.csv",
"scimagojr 2023  Subject Category - Psychology (miscellaneous).csv",
"scimagojr 2023  Subject Category - Psychiatry and Mental Health.csv",
"scimagojr 2023  Subject Category - Social Psychology.csv",
],

"developmentalBiology":
[
"scimagojr 2023  Subject Category - Developmental Biology.csv",
],

"ecologyEvolution":
[
"scimagojr 2023  Subject Category - Ecological Modeling.csv",
"scimagojr 2023  Subject Category - Ecology, Evolution, Behavior and Systematics.csv",
"scimagojr 2023  Subject Category - Ecology.csv",
],

"microbiology":
[
"scimagojr 2023  Subject Category - Microbiology (medical).csv",
"scimagojr 2023  Subject Category - Microbiology.csv",
"scimagojr 2023  Subject Category - Applied Microbiology and Biotechnology.csv",
"scimagojr 2023  Subject Category - Cell Biology.csv",
],

"molecularBiology":
[
"scimagojr 2023  Subject Category - Molecular Biology.csv",
"scimagojr 2023  Subject Category - Molecular Medicine.csv",
"scimagojr 2023  Subject Category - Structural Biology.csv",
"scimagojr 2023  Subject Category - Biochemistry, Genetics and Molecular Biology (miscellaneous).csv",
"scimagojr 2023  Subject Category - Biochemistry.csv",
"scimagojr 2023  Subject Category - Genetics (clinical).csv",
"scimagojr 2023  Subject Category - Genetics.csv",
],

"plantScience":
[
"scimagojr 2023  Subject Category - Plant Science.csv",
],

"varia":
[
"scimagojr 2023  Subject Category - Agricultural and Biological Sciences (miscellaneous).csv",
],
}


journals_by_field = dict()

for field in fields.keys():
# read journal list
    journals_list = set()
    for fname in fields[field]:
        journals = pd.read_csv(DATA_PATH+fname, quotechar='"', sep=";")
        journals_list.update(journals["Title"].str.lower().replace("[\(\[].*?[\)\]]", "", regex=True).str.strip())
    journals_by_field[field] = journals_list

# read metadata
metadata = []
for fos in FOS_LIST:
    _metadata = pd.read_csv(DATA_PATH+fos+"metadata.csv")
    _metadata["fos"] = fos
    metadata.append(_metadata)

metadata = pd.concat(metadata).rename(columns={"Unnamed: 0": "corpusid"})

metadata[list(journals_by_field.keys())] = pd.DataFrame(np.full((len(metadata.index),9), False), index=metadata.index)

for field in journals_by_field.keys():
    metadata[field] = metadata["journal"].str.lower().replace("[\(\[].*?[\)\]]", "", regex=True).str.strip().isin(journals_by_field[field])

# metadata = metadata[metadata[list(journals_by_field.keys())].sum(axis=1)==1]

metadata.to_csv(DATA_PATH+"fields_metadata.csv")
