from bert_serving.client import BertClient
from scipy import spatial
from itertools import combinations 
import numpy as np
import pandas as pd

# So far only looking at open ended data, ignoring questions 2-7

# OOB BERT, gotta run server and download pretrained model
bc = BertClient()
df = pd.read_excel(r"WesternAI's Matchorithm.xlsx")

def name_index(names, find):
	return list(names==find).index(True)

def year_pref(df, target, interest):
	names = df.iloc[:,1]
	t_pref_i = (1 - 0.2*abs(df.iloc[name_index(names, target), 3] - df.iloc[name_index(names, interest), 4]))**1.5
	i_pref_t = (1 - 0.2*abs(df.iloc[name_index(names, target), 4] - df.iloc[name_index(names, interest), 3]))**1.5
	res = (i_pref_t + t_pref_i)/2
	if i_pref_t!=i_pref_t and t_pref_i!=t_pref_i:
		return 0
	elif i_pref_t!=i_pref_t:
		return t_pref_i
	elif t_pref_i!=t_pref_i:
		return i_pref_t
	else:
		return res


def intensity_pref(df, target, interest):
	names = df.iloc[:,1]
	sim = (1 - 0.1*abs(df.iloc[name_index(names, target), 7] -  df.iloc[name_index(names, interest), 7]))**1.5
	return sim

def gender_pref(df, target, interest):
	names = df.iloc[:,1]
	t_pref_i = (df.iloc[name_index(names, target), 5] ==  df.iloc[name_index(names, interest), 6].rstrip("s"))
	i_pref_t = (df.iloc[name_index(names, target), 6].rstrip("s") == df.iloc[name_index(names, interest), 5])
	if t_pref_i and i_pref_t:
		return True
	else:
		return False

def sim(encodings):
	return (1 - spatial.distance.pdist(encodings, metric = 'cosine'))**2.5

# encodings = {}
# similarities = {}
sim_list = []
for col in df.iloc[:,8:13]:
	answers = bc.encode(list(df[col]))
	# encodings[col] = answers
	# similarities[col] = sim(answers)
	sim_list.append(sim(answers))

people = list(df.iloc[:,1])
couples = list(combinations(people, 2))

propensities = list(zip(*sim_list))

for i, c in enumerate(couples):
	propensities[i] = (*propensities[i], intensity_pref(df, c[0], c[1]), year_pref(df, c[0], c[1]))

results = [(*c, (sum(p)/len(p))**0.7) for c, p in zip(couples, propensities)]

matches = {}
for p in people:
	matches[p] = []

for r in results:
	matches[r[0]].append((r[1], r[2]))
	matches[r[1]].append((r[0], r[2]))

# Number of matches
n=2
for person, match in matches.items():
	match = [m for m in match if gender_pref(df, person, m[0])]
	matches[person] = sorted(match, key=lambda x: x[1], reverse=True)[0:(min(len(match), 2))]


for person, match in matches.items():
	print(df.iloc[people.index(person), 1])
	print(df.iloc[people.index(person), 2])
	print("Your matches:")
	for m in match:
		print("Matched ", m[0], " with ", int(100*m[1]), "% similarity!", sep="")
	print()

