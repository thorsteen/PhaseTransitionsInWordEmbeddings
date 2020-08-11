from matplotlib import pyplot as plt
import numpy as np

mean_rho_data = np.loadtxt("eval_mean_rho.txt", dtype = str, delimiter = "\n")
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

print("skip-gram evaluation WS-353 mean values, different context window sizes and versions")

#first results
#mean_rho_SGNS = [0.6356543304967629,0.6859671484411557  ,0.6924293781318873  , 0.7077708305390136,0.7188599859909027 , 0.7132944845703011 , 0.699551202038216, 0.7192855118852405,  0.7041803792159528,  0.717889605224073]
#window_size_SGNS =np.arange(1,len(mean_rho_SGNS)+1,1)

for i in range(len(mean_rho_SGNS)):
    print(model_name_SGNS[i] + ": " + str(mean_rho_SGNS[i]))

SGNS_plot, ax_SGNS = plt.subplots()
ax_SGNS.bar(np.arange(len(model_name_SGNS)), mean_rho_SGNS)
plt.xticks(np.arange(len(model_name_SGNS)), model_name_SGNS, rotation=90)
plt.yticks(np.arange(0, 0.8, step=0.1))
plt.gcf().subplots_adjust(bottom=0.6)
ax_SGNS.set_title("BlazingText SGNS models evaluation WS-353 mean rho values, different context window sizes and versions")
SGNS_plot.set_size_inches(11.69,3)
SGNS_plot.savefig("SGNS_plot.png")

#first results
#mean_rho_CBOW = [0.5804338343545967 , 0.6508858227099523 ,0.6802805346625579  ,0.6932546677573747  , 0.7024608832285866, 0.7090981595837526 ,0.7075484759236735, 0.722764153135901, 0.7121865820323352, 0.7234083268868999 ]
#window_size_CBOW = np.arange(1,len(mean_rho_CBOW)+1,1)

print("CBOW evaluation on WS-353 mean rho values, different context window sizes and versions")
for i in range(len(mean_rho_CBOW)):
    print(model_name_CBOW[i] + ": " + str(mean_rho_CBOW[i]))

CBOW_plot, ax_CBOW = plt.subplots()
ax_CBOW.bar(np.arange(len(model_name_CBOW)), mean_rho_CBOW)
plt.xticks(np.arange(len(model_name_CBOW)), model_name_CBOW, rotation=90)
plt.yticks(np.arange(0, 0.8, step=0.1))
plt.gcf().subplots_adjust(bottom=0.6)
ax_CBOW.set_title("BlazingText CBoW models evaluation WS-353 mean values, different context window sizes and versions")
CBOW_plot.set_size_inches(11.69,3)
CBOW_plot.savefig("CBOW_plot.png")
