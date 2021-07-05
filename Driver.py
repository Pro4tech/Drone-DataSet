#@Author:Pritesh Naik
import numpy as np    # import matplotlib.pyplot as plt
import adi			  # from matplotlib import transforms
import math 		  # import time
from scipy import signal
import pandas as pd
#Sample Min:300e3 Max:500e5
sample_rate = 250e4 # Hz
b,a =signal.butter(3,.2)
final = np.array([])
flist=np.array([])

##Function Calls
def filter(b,a,freq):
		z1=signal.lfilter_zi(b,a)
		pl=signal.lfilter(b,a,freq)
		return(pl)

def setup(center_freq,sample_rate,final):
	#PSD_Format
	def power(val):
		p=abs(val**2)
		pdb=math.log10(p)
		return(pdb)
	##SDR Setup Commands
	power=np.vectorize(power)
	fname = center_freq/1e9 
	name=str('{:.3f}Ghz'.format(fname))
	sdr = adi.Pluto("ip:192.168.2.1")
	sdr.sample_rate = int(sample_rate)
	sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
	sdr.rx_lo = int(center_freq)
	sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
	##Sampling Signal at Carrier Freq
	samples = sdr.rx()
	freqx=np.linspace(int(-sample_rate//2+center_freq),int(sample_rate//2+center_freq),int(1024))
	frq=np.fft.fft(samples)
	freq=np.fft.fftshift(frq)
	
	sig_avg = signal.find_peaks(filter(b,a,freq),height=10000) 	#Detecting Required Amplitude Peaks
	print('Size:',sig_avg[0].size)								#if(tuple!=empty)->then Adjust OffsetCorrection->then append in final[]
	temp=np.array(sig_avg[0])
	if(sig_avg[0].size>0):
		print("Active Band:{}".format(name))
		#temp=np.array(sig_avg[0])
		for i in range(0,temp.size):
			temp[i]=temp[i]*(sample_rate/1000)+center_freq
		print("Signal Peaks:",temp)
		final=np.concatenate((final, temp))
	return final
	

for i in range(0,10):
	con=str(i)	

	#Rx Function Codes
	final = np.array([])
	print("Sample rate:",sample_rate,'Hz')
	for center_freq in range(2405,2715,3):
		center_freq=int(center_freq*1e6)
		print("Center_freq:",center_freq,'Hz')
		final1=setup(center_freq,sample_rate,final)
		final=final1
	print("Final Peaks_Array:",final)
	store=pd.DataFrame(final)			
	store.to_csv("result.csv")

	#To get Non-(Wifi-Bluetooth) Signals
	#Importing .csv Files
	df1=pd.read_csv("result.csv",usecols=[1]) #Import Received Signal DataFrame
	df2=pd.read_csv("Non-DroneDataset.csv",usecols=[1]) #Importing Wifi-Bluetooth DataFrame

	#Converting DataFrames into numpyArray
	Dr= df1.to_numpy()
	Ndr = df2.to_numpy()

	#Check if Not Present in Non-Drone Set
	final=np.array([])
	for a in Dr:
  		a[0] = a[0].astype(int)
  		if(a not in Ndr):
    		test=a[0]
    		final=np.append(final,test)

	#To remove Duplicate Entries
	res = []
	for i in final:
    	if i not in res:
        	res.append(i)

	#Storing Results back into .csv format
	store=pd.DataFrame(res)
	store.to_csv(con+"Tx.csv",index=False)			#Input to Tx Code
	store=pd.DataFrame(Dr)
	store.to_csv("Rx.csv")			#Storage of Received Signals

print("End Code")


