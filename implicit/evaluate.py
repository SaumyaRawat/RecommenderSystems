from scipy import stats
import numpy as np
import pandas as pd

#Read output
def read_output(filename,header_col):
	data = pd.read_table(filename, usecols=[0, 1, 2, 3], names=header_col, delim_whitespace=True)
	return data
 
#Calculate percentile rank for an array of n items
def rank_ui(n):
	factor = 100/n
	for i in range(n-1):
		ranks.append(i*factor)
	ranks.append(100)

		ranks.append(i*(100/len()))
if __name__ == "__main__":
	
	filename = 'similar-userIDs.tsv'
	header_col = ['user_item','rec','conf_score','sim_score']
	data = read_output(filename,header_col)
	
	#get ranked list row from data
	rec_list = [[]]
	ind_list_of_rec = []
	grouped = data.groupby('user_item')
	mpr_scores.apply(sum(data.conf_score*rank_ui(data.conf_score.size())))
	mpr = mpr_scores/data.conf_score.sum()
	#for index, row in data.iterrows():
	#	start = 0 
	#	if index%k==0:
	#		rec_list.append(ind_list_of_rec) #append the prev list of ranked items
	#		ind_list_of_rec = []
	#	ind_list_of_rec.append(row['sim_score'])
	#rec_list.append(ind_list_of_rec) # append the last row left

	#rec_list = [a for a in rec_list if a != []]
	print(mpr)
	
	#for x in rec_list:
	#	print(np.percentile(np.array(x),1,axis=0))
	#	pos = [stats.percentileofscore(x, i) for i in x]
	#	print(pos)