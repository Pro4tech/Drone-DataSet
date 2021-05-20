import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib import transforms
import time
import math 
from scipy import signal
import pandas as pd
#Sample Min:300e3 Max:500e5
sample_rate = 250e4 # Hz
b,a =signal.butter(3,.2)
#final = np.array([])
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
	##Plotting Frequency Response			#freq1=power(freq)
	plt.plot(freqx,freq)						#plt.plot(freqxfreq)
	plt.xlabel('Frequency(Base Freq+Offset)Hz')
	plt.ylabel('Amplitude of Signal')
	plt.title("Frequency Amplitude:"+name) #for Single TimeFrame
	plt.savefig("Clean"+name+"-FA.png")				#/////
	#plt.show()
	plt.close()	
	##Storing Dataframe in .csv format
	store=pd.DataFrame(freq)									# Storing Dataframe using PANDAS -> IMPORT TO TX IN PANDAS
	store.to_csv("Clean"+name+"-Dataframe.csv")
	
	# #issue:Relative plot need abs plot spectogram

	samplegroup=[]
	for _ in range(1000):
		samples=sdr.rx()
		frq=np.fft.fft(samples)
		freq=np.fft.fftshift(frq)
		#freq=power(freq)
		samplegroup.append(abs(freq))
	plt.imshow(samplegroup)
	plt.set_cmap('hot')
	plt.xlabel('Frequency(Base Freq+offset)Hz')
	plt.title("Waterfall Diagram:"+name) #for 1000ms TimeFrame
	plt.savefig("Clean"+name+"-WD.png")
	plt.close
	samples=sdr.rx()
	f, t, Sxx = signal.spectrogram(samples, sample_rate,scaling='spectrum', axis=- 1, mode='psd')
	base=plt.gca().transData
	rot=transforms.Affine2D().rotate_deg(-90)
	plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Sxx, axes=0), shading='gouraud',transform=rot+base)
	plt.pcolormesh(t, f, Sxx, shading='gouraud', transform=rot+base)

	plt.xlabel('Frequency [Hz]')

	plt.ylabel('Time [sec]')

	#plt.show()
	plt.savefig("Clean"+name+"-spectrum.png") 	#///
	# plt.show()
	plt.close()
	return final

##Main Function Codes
print("Sample rate:",sample_rate,'Hz')
#rangeF=[2475,2435,]
for i in range(0,100):
	final = np.array([])
	for center_freq in range(2405,2715,3):
		center_freq=int(center_freq*1e6)
		print("Center_freq:",center_freq,'Hz')
		final1=setup(center_freq,sample_rate,final)
		final=final1
	print("Final Peaks_Array:",final)
	store=pd.DataFrame(final)
	con=str(i)
	store.to_csv("Peak-"+con+".csv")
