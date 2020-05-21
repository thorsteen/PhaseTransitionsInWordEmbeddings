from matplotlib import pyplot as plt
import numpy as np

mean_rho_data = np.loadtxt("eval_mean_rho.txt", dtype = str, delimiter = "\n")
#we know there shoould be 12 values in both
mean_rho_SGNS = []
mean_rho_CBOW = []
model_name_SGNS = []
model_name_CBOW = []

for i, data in enumerate(mean_rho_data):
    if i == len(mean_rho_data)-1:
        print("made arrays of values")
    else:
        if "cbow" in str(data):
            model_name_CBOW.append(mean_rho_data[i])
            mean_rho_CBOW.append(float(mean_rho_data[i+1]))
        if "skipgram" in str(data):
            model_name_SGNS.append(mean_rho_data[i])
            mean_rho_SGNS.append(float(mean_rho_data[i+1]))

mean_rho_SGNS = np.array(mean_rho_SGNS)
mean_rho_CBOW = np.array(mean_rho_CBOW)
#mean_rho_SGNS.sort() #we fuck it up if we sort
#mean_rho_CBOW.sort() #could make tuples instead...

print("skip-gram evaluation WS-353 mean values, windows size 1 - {}".format(len(mean_rho_SGNS)))

#first results
#mean_rho_SGNS = [0.6356543304967629,0.6859671484411557  ,0.6924293781318873  , 0.7077708305390136,0.7188599859909027 , 0.7132944845703011 , 0.699551202038216, 0.7192855118852405,  0.7041803792159528,  0.717889605224073]
#window_size_SGNS =np.arange(1,len(mean_rho_SGNS)+1,1)

for i in range(len(mean_rho_SGNS)):
    print(model_name_SGNS[i] + ": " + str(mean_rho_SGNS[i]))

SGNS_plot, ax_SGNS = plt.subplots()
ax_SGNS.plot(model_name_SGNS, mean_rho_SGNS)
plt.xticks(np.arange(len(model_name_SGNS)), model_name_SGNS, rotation=20)
ax_SGNS.set_title("skip-gram evaluation WS-353 mean rho values, windows size 1 - {}".format(len(mean_rho_SGNS)))
SGNS_plot.savefig("SGNS_plot.png")

#first results
#mean_rho_CBOW = [0.5804338343545967 , 0.6508858227099523 ,0.6802805346625579  ,0.6932546677573747  , 0.7024608832285866, 0.7090981595837526 ,0.7075484759236735, 0.722764153135901, 0.7121865820323352, 0.7234083268868999 ]
#window_size_CBOW = np.arange(1,len(mean_rho_CBOW)+1,1)

print("CBOW evaluation on WS-353 mean rho values, windows size 1 - {}".format(len(mean_rho_CBOW)))
for i in range(len(mean_rho_CBOW)):
    print(model_name_CBOW[i] + ": " + str(mean_rho_CBOW[i]))

CBOW_plot, ax_CBOW = plt.subplots()
ax_CBOW.plot(model_name_CBOW, mean_rho_CBOW)
plt.xticks(np.arange(len(model_name_CBOW)), model_name_CBOW, rotation=20)
ax_CBOW.set_title("CBOW evaluation WS-353 mean values, windows size 1 - {}".format(len(mean_rho_CBOW)))
CBOW_plot.savefig("CBOW_plot.png")
