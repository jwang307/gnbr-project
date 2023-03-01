from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

relation_lookup = {
    'T': "treatment",
    'C': "inhibits cell growth",
    'Sa': "side effect",
    'Pr': "prevents",
    'Pa': "alleviates",
    'J': "role in disease pathogenesis",
}

# Part I files
part_i_files = ['part-i-chemical-disease-path-theme-distributions 2.txt']
part_ii_files = ['part-ii-dependency-paths-chemical-disease-sorted-with-themes.txt']

# Process part I files
for file in part_i_files:
    path = Path('./cache/processed_{}.p'.format(file.removesuffix(".txt")))

    if path.exists():
        print("File {} already processed".format(file))
        continue

    master_dict = {}

    with open("./files/{}".format(file), 'r') as f:
        headers = f.readline().strip('\n').split('\t')
        for line in tqdm.tqdm(f):
            split = line.strip('\n').split('\t')
            assert len(headers) == len(split)

            # dependency path
            path = split[0]

            scores = split[1:]
            theme = headers[np.argmax(scores) + 1]

            master_dict[path] = theme

    # Cache so we dont have to do this again
    with open('./cache/processed_{}.p'.format(file.removesuffix(".txt")), 'wb') as fp:
        pickle.dump(master_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Process part II files
for file in part_ii_files:
    path = Path('./cache/processed_{}.p'.format(file.removesuffix(".txt")))

    if path.exists():
        print("File {} already processed".format(file))
        continue

    master_array = []

    with open("./files/{}".format(file), 'r') as f:
        headers = ['PubMed ID','Sentence number (0 = title)','First entity name, formatted','First entity name, location (characters from start of abstract)','Second entity name, formatted','Second entity name, location','First entity name, raw string','Second entity name, raw string','First entity name, database ID(s)','Second entity name, database ID(s)','First entity type (Chemical, Gene, Disease)','Second entity type (Chemical, Gene, Disease)','Dependency path','Sentence, tokenized']
        for lineno,line in tqdm.tqdm(enumerate(f)):
            split = line.strip('\n').split('\t')
            assert len(headers) == len(split)

            row = { h:v for h,v in zip(headers,split) }

            text1 = row['First entity name, formatted']
            text2 = row['Second entity name, formatted']
            path = row['Dependency path'].lower()

            master_array.append([text1, text2, path])

    # Cache so we dont have to do this again
    with open('./cache/processed_{}.p'.format(file.removesuffix(".txt")), 'wb') as fp:
        pickle.dump(master_array, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Generate triples
part_i_and_ii_files = [ (file1, file2) for file1, file2 in zip(part_i_files, part_ii_files) ]

for pair in part_i_and_ii_files:
    cached_file_part_i = './cache/processed_{}.p'.format(pair[0].removesuffix(".txt"))
    cached_file_part_ii = './cache/processed_{}.p'.format(pair[1].removesuffix(".txt"))

    with open(cached_file_part_i, 'rb') as f1p, open(cached_file_part_ii, 'rb') as f2p:
        path_to_theme = pickle.load(f1p)
        entities_and_paths = pickle.load(f2p)

        # TODO: Generalize the following two lines when we have more themes
        chemicals = []
        diseases = []

        triples = []

        for entity1, entity2, path in entities_and_paths:
            theme = path_to_theme[path]

            # Skip these themes for now
            if theme == "Mp":
                continue

            triple = (entity1, relation_lookup[theme], entity2)
            triples.append(triple)
            chemicals.append(entity1)
            diseases.append(entity2)

        # Write array to tsv
        df = pd.DataFrame(list(set(triples)), columns=['entity1', 'theme', 'entity2'])
        df.to_csv('./triples/unique_{}.tsv'.format(pair[1].replace("part-ii-dependency-paths-", "").replace("-sorted-with-themes.txt", "")), sep='\t', index=False)

        # df = pd.DataFrame(triples, columns=['entity1', 'theme', 'entity2'])
        # df.to_csv('./triples/non_unique_{}.tsv'.format(pair[1].replace("part-ii-dependency-paths-", "").replace("-sorted-with-themes.txt", "")), sep='\t', index=False)


        # TODO: Generalize the following lines when we have more themes
        # Write entities to tsv
        df = pd.DataFrame(list(set(chemicals)), columns=['chemical'])
        df.to_csv('./entities/chemicals.tsv', sep='\t', index=False)

        df = pd.DataFrame(list(set(diseases)), columns=['disease'])
        df.to_csv('./entities/diseases.tsv', sep='\t', index=False)
