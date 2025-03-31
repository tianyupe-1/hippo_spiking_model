import nest
from scipy import io
import scipy.io.matlab
import scipy.signal
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

tstop = 5 # simulation time (s)
scaleing_TM = 10 # the scaling of synaptic weights in TM model of STP

def setNestOpts():
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.local_num_threads = 7

def rasterPlot(ev):
    srcs = ev['senders']
    srcs_ = set(ev['senders'])
    m = min(srcs_)
    M = max(srcs_)
    tims = ev['times']
    for s in srcs_:
        c = f'#2757{int((s-m)/(M-m)*255):02X}'
        plt.vlines(tims[srcs == s], s-m, s-m+1, c)


###
# Synapte models
###

def static_syn(w): # static synapse
    return {
      'weight': w,
      'delay': 1 #1
    }

def plastic_syn(w): # TM model of STP
    return {
        'synapse_model': 'tsodyks2_synapse',
        'U': 0.5, # 0.5; real | Parameter determining the increase in u
        'u': 0.35, # 0.4; The probability of release (U_se) [0,1], default=0.5
        'tau_fac': 1e2, # 1e2; ms | Time constant for facilitation
        'tau_rec': 1e2, # 1e2; ms | Time constant for recovery
        'x': 1, #1, real | Scale the amount of neurotransmitter release
        'weight': w*scaleing_TM, # scaling para based on synaptic connectivity strength w
        'delay': 1,
    }


###
# synaptic connectivity weight matrix
   ## 1:PYR; 2:BiC; 3:PV; 4:CCK; 
   ## w_ij means the connectivity strength of the projection from neuron j to neuron i
###
   

# From PYR (excitatory)
w11 = 0.035
w21 = 0.035
w31 = 0.035
w41 = 0.035

# From BiC (inhibitory)
w12 = 0.1
w22 = 0.1
w32 = 0.1
w42 = 0.1

# From PV (inhibitory)
w13 = 0.1
w23 = 0.1
w33 = 0.1
w43 = 0.1

# From CCK (inhibitory)
w14 = 0.1
w24 = 0.1
w34 = 0.1
w44 = 0.1

weights = [[w11,-w12,-w13,-w14],
               [w21,-w22,-w23,-w24],
               [w31,-w32,-w33,-w34],
               [w41,-w42,-w43,-w44]]


###
# The main model simulation function
###

def runSim(weights, tstop, exc_static=False, inh_static=False): # decide if excitatory or inhibitory synapse is static

   # tstop = 5 # simulation time (s)

    setNestOpts()

    ##
    # Neuronal groups
    ##
    
    # number of neurons for each group
    PYR_n = 300
    BiC_n = 20 
    PV_n = 20
    CCK_n = 20 

    # injection currents (STIM) to each neuronal group
    PYR_I = 1 
    BiC_I = 2 
    PV_I = 1 
    CCK_I = 1 
    
    ##
    # Define synapses
    ##

    def syn(w, excOrInh): # plastic (TM) or static synapses
        isStatic = None
        if excOrInh == "exc":
            isStatic = exc_static
        else:
            isStatic = inh_static
        if isStatic:
            return static_syn(w)
        else:
            return plastic_syn(w)

    # sparse connectivity probabilities
    conn_prob_PYR = 0.1 # proportion of PYR neurons projecting to a certain neuron
    conn_prob_BiC = 0.25 # proportion of BiC neurons projecting to a certain neuron
    conn_prob_PV = 0.25 # proportion of PV neurons projecting to a certain neuron
    conn_prob_CCK = 0.25 # proportion of CCK neurons projecting to a certain neuron
    
    # NEST connectivity setup
    conn_dict_PYR = {"rule": "fixed_indegree","indegree": int(conn_prob_PYR*PYR_n)}
    conn_dict_BiC = {"rule": "fixed_indegree","indegree": int(conn_prob_BiC*BiC_n)}
    conn_dict_PV = {"rule": "fixed_indegree","indegree": int(conn_prob_PV*PV_n)}
    conn_dict_CCK = {"rule": "fixed_indegree","indegree": int(conn_prob_CCK*CCK_n)}

    #def sparse_conn(src, dst, conn_dict, w, excOrInh):
        #nest.Connect(src, dst, conn_dict, syn_spec=syn(w, excOrInh))
        
    all_syn_parrots = []    
    def sparse_conn(src, dst, conn_dict, w, excOrInh): # incorporate randomness 
        n_src = len(src)
        n_nodes = conn_dict["indegree"]
        for d in dst:
            src_i = np.random.choice(np.arange(n_src),n_nodes,replace=False)
            src_i.sort()
            src_ = src[src_i]
            prts = nest.Create("parrot_neuron",n_nodes)
            all_syn_parrots.extend(prts.tolist())
            nest.Connect(src_, prts, 'one_to_one') 
            nest.Connect(prts, d, syn_spec=syn(w, excOrInh))  

    ##
    # PSTH computing method setup
    ##

    bin_w = 1 # ms; PSTH bin width
    bin_edge = np.arange(0,tstop*1000+bin_w,bin_w) # all bins
    sigma = 10 # 5 ms, SD of the Gaussian kernel for computing PSTH
    s3 = 3*sigma
    gx = np.arange(-s3, s3, bin_w)
    PSTH_window = np.exp(-((gx/sigma)**2)/2)/((2*np.pi*sigma**2)**0.5) #PSTH Gaussian kernel
    time_vec = np.arange(1,tstop*1000+bin_w,bin_w) # time_vec for plot

    ###
    # Simulation
    ###

    ##
    # Izhi model
    ##

    PYR = nest.Create("izhikevich", PYR_n, params={
        "V_th": nest.random.normal(30, PYR_I/2),
        "a": 0.02,
        "b": 0.25,
        "c": -65,
        "d": 0.05
    })

    PYR_bias = nest.Create("noise_generator", params={
        "mean": PYR_I,
        "std": PYR_I/2 #PYR_I/2
    })

    nest.Connect(PYR_bias, PYR)


    BiC = nest.Create("izhikevich", BiC_n, params={
        "V_th": nest.random.normal(30, BiC_I/2),
        "a": 0.015,
        "b": 0.25, #0.2
        "c": -65,
        "d": 2.05
    })

    BiC_bias = nest.Create("noise_generator", params={
        "mean": BiC_I,
        "std": BiC_I/2
    })

    nest.Connect(BiC_bias, BiC)


    PV = nest.Create("izhikevich", PV_n, params={
        "V_th": nest.random.normal(30, abs(PV_I)/2),
        "a": 0.015, #0.02; 0.015
        "b": 0.25, #0.25
        "c": -65,
        "d": 2.05 # 2.05
    })

    PV_bias = nest.Create("noise_generator", params={
        "mean": PV_I,
        "std": abs(PV_I)/2
    })

    nest.Connect(PV_bias, PV)
    
    CCK = nest.Create("izhikevich", CCK_n, params={
        "V_th": nest.random.normal(30, abs(CCK_I)/2),
        "a": 0.015, #0.02; 0.015
        "b": 0.25, #0.25
        "c": -65,
        "d": 2.05 # 2.05
    })

    CCK_bias = nest.Create("noise_generator", params={
        "mean": CCK_I,
        "std": abs(CCK_I)/2
    })

    nest.Connect(CCK_bias, CCK)    

    ##
    # Connection specifications (sparse)
    ##

    # 1. PYR
    # PYR->PYR
    sparse_conn(PYR, PYR, conn_dict_PYR, weights[0][0], "exc")
    # PYR->BiC
    sparse_conn(PYR, BiC, conn_dict_PYR, weights[1][0], "exc")
    # PYR->PV
    sparse_conn(PYR, PV, conn_dict_PYR, weights[2][0], "exc")
    # PYR->CCK
    sparse_conn(PYR, CCK, conn_dict_PYR, weights[3][0], "exc")    

    # 2. BiC
    # BiC->PYR
    sparse_conn(BiC, PYR, conn_dict_BiC, weights[0][1], "inh")    
    # BiC->BiC
    sparse_conn(BiC, BiC, conn_dict_BiC, weights[1][1], "inh")
    # BiC->PV
    sparse_conn(BiC, PV, conn_dict_BiC,  weights[2][1], "inh")
    # BiC->CCK
    sparse_conn(BiC, CCK, conn_dict_BiC, weights[3][1], "inh")    

    # 3. PV
    # PV->PYR
    sparse_conn(PV, PYR, conn_dict_PV, weights[0][2], "inh")
    # PV->BiC
    sparse_conn(PV, BiC, conn_dict_PV, weights[1][2], "inh")
    # PV->PV
    sparse_conn(PV, PV, conn_dict_PV, weights[2][2], "inh")    
    # PV->CCK
    sparse_conn(PV, CCK, conn_dict_PV, weights[3][2], "inh")    
    

    # 4. CCK
    # CCK->PYR
    sparse_conn(CCK, PYR, conn_dict_CCK, weights[0][3], "inh")
    # CCK->BiC
    sparse_conn(CCK, BiC, conn_dict_CCK, weights[1][3], "inh")
    # CCK->PV
    sparse_conn(CCK, PV, conn_dict_CCK, weights[2][3], "inh")    
    # CCK->CCK
    sparse_conn(CCK, CCK, conn_dict_CCK, weights[3][3], "inh")         
    

    ##
    # Recording
    ##

    PYR_spikes = nest.Create("spike_recorder")
    nest.Connect(PYR, PYR_spikes)

    BiC_spikes = nest.Create("spike_recorder")
    nest.Connect(BiC, BiC_spikes)

    PV_spikes = nest.Create("spike_recorder")
    nest.Connect(PV, PV_spikes)
    
    CCK_spikes = nest.Create("spike_recorder")
    nest.Connect(CCK, CCK_spikes)    


    ##
    # Simulation 
    ##

    nest.Simulate(tstop*1000)

    PYR_ev = PYR_spikes.get("events")
    BiC_ev = BiC_spikes.get("events")
    PV_ev = PV_spikes.get("events")
    CCK_ev = CCK_spikes.get("events")
    
    ## average firing rate
    print(f"PYR firing rate: {spikeStats(PYR_ev,tstop, PYR_n)} Hz")
    print(f"BiC firing rate: {spikeStats(BiC_ev,tstop, BiC_n)} Hz")
    print(f"PV firing rate: {spikeStats(PV_ev,tstop, PV_n)} Hz")
    print(f"CCK firing rate: {spikeStats(CCK_ev,tstop, CCK_n)} Hz")
    
    ## instantaneuous firing rate (PSTH)
    
    # PYR
    PYR_ts = PYR_ev['times']
    PYR_hist = np.histogram(PYR_ts,bins=bin_edge)[0]*1000/PYR_n
    kernel_PSTH_PYR_ori = scipy.signal.fftconvolve(PYR_hist,PSTH_window,mode='same')    
    mean_FR_PYR = (len(PYR_ev['times'])/PYR_n)/tstop # scale by the mean FR
    scale = mean_FR_PYR/np.mean(kernel_PSTH_PYR_ori)
    kernel_PSTH_PYR = scale*kernel_PSTH_PYR_ori
    
    # BiC
    BiC_ts = BiC_ev['times']
    BiC_hist = np.histogram(BiC_ts,bins=bin_edge)[0]*1000/BiC_n
    kernel_PSTH_BiC_ori = scipy.signal.fftconvolve(BiC_hist,PSTH_window,mode='same')    
    mean_FR_BiC = (len(BiC_ev['times'])/BiC_n)/tstop # scale by the mean FR
    scale = mean_FR_BiC/np.mean(kernel_PSTH_BiC_ori)
    kernel_PSTH_BiC = scale*kernel_PSTH_BiC_ori   
    
    # PV
    PV_ts = PV_ev['times']
    PV_hist = np.histogram(PV_ts,bins=bin_edge)[0]*1000/PV_n
    kernel_PSTH_PV_ori = scipy.signal.fftconvolve(PV_hist,PSTH_window,mode='same')    
    mean_FR_PV = (len(PV_ev['times'])/PV_n)/tstop # scale by the mean FR
    scale = mean_FR_PV/np.mean(kernel_PSTH_PV_ori)
    kernel_PSTH_PV = scale*kernel_PSTH_PV_ori  
    
    # CCK
    CCK_ts = CCK_ev['times']
    CCK_hist = np.histogram(CCK_ts,bins=bin_edge)[0]*1000/PV_n
    kernel_PSTH_CCK_ori = scipy.signal.fftconvolve(CCK_hist,PSTH_window,mode='same')    
    mean_FR_CCK = (len(CCK_ev['times'])/CCK_n)/tstop # scale by the mean FR
    scale = mean_FR_CCK/np.mean(kernel_PSTH_CCK_ori)
    kernel_PSTH_CCK = scale*kernel_PSTH_CCK_ori     
        

    return time_vec, kernel_PSTH_PYR, kernel_PSTH_BiC, kernel_PSTH_PV, kernel_PSTH_CCK,PYR_ev,BiC_ev,PV_ev,CCK_ev

###
# outputs and plots
###

def plot_FR(t, FR, color_sig, cell_type):
    plt.plot(t, FR, color = color_sig, label= cell_type)
    plt.legend()
    plt.title(f"firing rate - {cell_type}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")

def spikeStats(ev, time_len, n=1): # compute mean FR of a sample neuron in a neuronal group
    srcs = ev['senders']
    means = np.bincount(srcs)[-n:]/time_len
    return np.mean(means)


time_vec, kernel_PSTH_PYR, kernel_PSTH_BiC, kernel_PSTH_PV, kernel_PSTH_CCK,PYR_ev,BiC_ev,PV_ev,CCK_ev = runSim(weights,tstop, False, False)


# PSTH firing rate plots

plt.figure(figsize=(16,12))
plt.subplot(221)
plot_FR(time_vec, kernel_PSTH_PYR, 'r', 'PYR')
plt.xlabel("Time (ms)")
plt.subplot(222)
plot_FR(time_vec, kernel_PSTH_BiC, 'b', 'BiC')
plt.xlabel("Time (ms)")
plt.subplot(223)
plot_FR(time_vec, kernel_PSTH_PV, 'y', 'PV')
plt.xlabel("Time (ms)")
plt.subplot(224)
plot_FR(time_vec, kernel_PSTH_CCK, 'k', 'CCK')
plt.xlabel("Time (ms)")
plt.show()

io.savemat(f"FR_PYR.mat", {f"FR_PYR":kernel_PSTH_PYR})
io.savemat(f"FR_BiC.mat", {f"FR_BiC":kernel_PSTH_BiC})
io.savemat(f"FR_PV.mat", {f"FR_PV":kernel_PSTH_PV})
io.savemat(f"FR_CCK.mat", {f"FR_CCK":kernel_PSTH_CCK})

# raster plots

plt.figure(figsize=(16,12))
plt.subplot(221)
rasterPlot(PYR_ev)
plt.ylabel("PYR neuron id")
plt.xlabel("Time (ms)")
plt.subplot(222)
rasterPlot(BiC_ev)
plt.ylabel("BiC neuron id")
plt.xlabel("Time (ms)")
plt.subplot(223)
rasterPlot(PV_ev)
plt.ylabel("PV neuron id")
plt.xlabel("Time (ms)")
plt.subplot(224)
rasterPlot(CCK_ev)
plt.ylabel("CCK neuron id")
plt.xlabel("Time (ms)")
#plt.savefig(f"img/Raster.pdf")
plt.show()



