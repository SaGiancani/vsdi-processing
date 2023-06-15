# Author: Philippe Katz <philippe.katz@gmail.com>,
#         Sylvain Takerkart <Sylvain.Takerkart@incm.cnrs-mrs.fr>
# License: BSD Style.
#
# Modified by Isabelle Racicot <racicot.isabelle@gmail.com> on 12/2019 
# Python3 portability by Salvatore Giancani <sa.giancani@gmail.com>
import cv2 as cv
import itertools as it
import io
import numpy as np
import struct

# Commit Try
class BlkFile:
	"""This class contains some methods to load data from BLK file and write it into NIFTI format
	Attributes
	----------
	header : dict
	    The header of the input BLK File
	data : numpy array
	    The data of the input BLK File without transformation
	image : numpy array
	    The data after transformation, in a 4D image format
	p : str
	    len of string
	h : str
	    len of short
	H : str
	    len of unsigned short
	i : str
	    len of integer
	f : str
	    len of float
	l : str
	    len of long
	filename : str
	    The path of the input BLK file
	Methods
	-------
	__init__( ... )
	    Initializes attributes
	sizeofunity( ... )
	    Adaptation of data type size
	get_head( ... )
	    Reads each field of BLK file's header and save them into a dictionnary
	get_data( ... )
	    Reads the data contained in the BLK file
	get_4d_image( ... )
	    Reads the data contained in the BLK file
	"""

	def __init__(self, filename, spatial_binning, temporal_binning, header = None, motion_switch = False, filename_particle = 'vsd_C'):
		"""Initializes attributes
		Default values for:
		* p : '1p'
		* h : '2h'
		* H : '2H'
		* i : '4i'
		* f : '4f'
		* l : '8l'
		Parameters
		----------
		filename : str
		    The path of the external file, containing the raw image
		"""
		# Adaptation of data type size
		self.p = self.sizeofunity(1,'p')
		self.h = self.sizeofunity(2,'h')
		self.H = self.sizeofunity(2,'H')
		self.i = self.sizeofunity(4,'i')
		self.I = self.sizeofunity(4,'I')
		self.f = self.sizeofunity(4,'f')
		self.l = self.sizeofunity(8,'l')
		self.L = self.sizeofunity(8,'L')
		self.filename=filename # String of filename
		if header is None:
			self.header=self.get_head() # Dictionary for metadata
		else:
			self.header = header
		self.data=self.get_data() # Raw vsdi data
		self.signal=self.get_signal() # Images for vsdi
		#self.image=self.get_image(detrend) # Images for vsdi
		self.spatial_binning = spatial_binning # Binning value for space (both x and y)
		self.temporal_binning = temporal_binning # Binning value for time
		self.binned_signal = self.bin_signal()#bin_image(interpolation_mode = cv.INTER_CUBIC) #default is INTER_LINEAR	
		self.condition = int(self.filename.split(filename_particle)[1][0:2]) #Adding an extracting string directly from filename
		if motion_switch:
			self.motion_ind, self.motion_ind_max = self.motion_index() # Motion index: float
		#self.df_fz = process.deltaf_up_fzero(self.binned_image, self.zero_frames, deblank = dblnk, blank_sign = blank_signal)
		#self.time_course_sign = self.get_tc_signal(roi_mask = roi_mask) # Instantiation of dtrnd_roi_sign. numpy.array 1D
		#self.roi_sign = self.get_roi_signal(roi_mask = roi_mask) # Instantiation of dtrnd_roi_sign. numpy.array 1D
		#self.select_flag = False # Boolean flag for autoselection of trial/blk


	def sizeofunity(self,size,string):
		"""Adaptation of data type size
		
		Every systems don't has the same data type size. This function makes an adaptation of this size to let the script compatible on every system.
		Parameters
		----------
		size : int
		    The size of the wanted unity
		string : {'p','h','H','i','f','l'}
		    'p' : char[],
		    'h' : short,
		    'H' : unsigned short,
		    'i' : integer,
		    'f' : float,
		    'l' : long,
		Returns
		-------
		size_of_unity : str
		    A concatenation of data type and the type size evaluted on every system
		"""
		unit = str(size//struct.calcsize(string)) # The wanted size of data type divided by the data type size of the system

		size_of_unity = unit+string # Concatenation of unit and data type name
		return size_of_unity

	def get_head(self):
		"""Reads each field of BLK file's header and save them into a dictionnary
		Returns
		-------
		header : dict
		    The head of the BLK file, containings meta-datas
		"""
		from struct import unpack

		fid = io.open(self.filename,'rb')

		filesize = fid.read(8)
		header = {}
		header['filesize'] = unpack(self.l,filesize)[0]

		checksum_header = fid.read(2)
		header['checksum_header'] = unpack(self.h,checksum_header)[0]

		checksum_data= fid.read(2)
		header['checksum_data'] = unpack(self.h,checksum_data)[0]

		lenheader = fid.read(4)
		header['lenheader'] = unpack(self.i,lenheader)[0]

		versionid = fid.read(4)
		header['versionid'] = unpack(self.f,versionid)[0]

		filetype = fid.read(4)
		header['filetype'] = unpack(self.i,filetype)[0]
        		# RAWBLOCK_FILE          (11)
		        # DCBLOCK_FILE           (12)
	        	# SUM_FILE               (13)

		filesubtype = fid.read(4)
		header['filesubtype'] = unpack(self.i,filesubtype)[0]
		        # FROM_VDAQ              (11)
		        # FROM_ORA               (12)
		        # FROM_DYEDAQ            (13)

		datatype = fid.read(4)
		header['datatype'] = unpack(self.i,datatype)[0]
		        # DAT_UCHAR     (11)
		        # DAT_USHORT    (12)
		        # DAT_LONG      (13)
		        # DAT_FLOAT     (14)

		sizeof = fid.read(4)
		header['sizeof'] = unpack(self.i,sizeof)[0]
		        # e.g. sizeof(long), sizeof(float)

		framewidth = fid.read(4)
		header['framewidth'] = unpack(self.i,framewidth)[0]

		frameheight = fid.read(4)
		header['frameheight'] = unpack(self.i,frameheight)[0]

		nframesperstim = fid.read(4)
		header['nframesperstim'] = unpack(self.i,nframesperstim)[0]

		nstimuli = fid.read(4)
		header['nstimuli'] = unpack(self.i,nstimuli)[0]

		initialxbinfactor = fid.read(4)
		header['initialxbinfactor'] = unpack(self.i,initialxbinfactor)[0]
        		# from data acquisition

		initialybinfactor = fid.read(4)
		header['initialybinfactor'] = unpack(self.i,initialybinfactor)[0]
        		# from data acquisition

		xbinfactor = fid.read(4)
		header['xbinfactor'] = unpack(self.i,xbinfactor)[0]
        		# this file

		ybinfactor = fid.read(4)
		header['ybinfactor'] = unpack(self.i,ybinfactor)[0]
        		# this file

		header['username'] = fid.read(32)

		recordingdate = fid.read(16)
		header['recordingdate'] = unpack(str(int(self.p[0:-1])*16)+self.p[-1],recordingdate)

		x1roi = fid.read(4)
		header['x1roi'] = unpack(self.i,x1roi)[0]

		y1roi = fid.read(4)
		header['y1roi'] = unpack(self.i,y1roi)[0]

		x2roi = fid.read(4)
		header['x2roi'] = unpack(self.i,x2roi)[0]

		y2roi = fid.read(4)
		header['y2roi'] = unpack(self.i,y2roi)[0]

		stimoffs = fid.read(4)
		header['stimoffs'] = unpack(self.i,stimoffs)[0]

		stimsize = fid.read(4)
		header['stimsize'] = unpack(self.i,stimsize)[0]

		frameoffs = fid.read(4)
		header['frameoffs'] = unpack(self.i,frameoffs)[0]

		framesize = fid.read(4)
		header['framesize'] = unpack(self.i,framesize)[0]

		refoffs = fid.read(4)
		header['refoffs'] = unpack(self.i,refoffs)[0]

		refsize = fid.read(4)
		header['refsize'] = unpack(self.i,refsize)[0]

		refwidth = fid.read(4)
		header['refwidth'] = unpack(self.i,refwidth)[0]

		refheight = fid.read(4)
		header['refheight'] = unpack(self.i,refheight)[0]

		whichblocks = fid.read(2*16)
		header['whichblocks'] = unpack(str(int(self.H[0:-1])*16)+self.H[-1],whichblocks)[0]

		whichframes = fid.read(2*16)
		header['whichframes'] = unpack(str(int(self.H[0:-1])*16)+self.H[-1],whichframes)[0]

		# DATA ANALYSIS
		loclip = fid.read(4)
		header['loclip'] = unpack(self.i,loclip)[0]

		hiclip = fid.read(4)
		header['hiclip'] = unpack(self.i,hiclip)[0]

		lopass = fid.read(4)
		header['lopass'] = unpack(self.i,lopass)[0]

		hipass = fid.read(4)
		header['hipass'] = unpack(self.i,hipass)[0]

		header['operationsperformed'] = fid.read(64)

		# ORA-SPECIFIC
		magnification = fid.read(4)
		header['magnification'] = unpack(self.f,magnification)[0]

		gain = fid.read(2)
		header['gain'] = unpack(self.H,gain)[0]

		wavelength = fid.read(2)
		header['wavelength'] = unpack(self.H,wavelength)[0]

		exposuretime = fid.read(4)
		header['exposuretime'] = unpack(self.i,exposuretime)[0]

		nrepetitions = fid.read(4)
		header['nrepetitions'] = unpack(self.i,nrepetitions)[0]

		acquisitiondelay = fid.read(4)
		header['acquisitiondelay'] = unpack(self.i,acquisitiondelay)[0]

		interstiminterval = fid.read(4)
		header['interstiminterval'] = unpack(self.i,interstiminterval)[0]

		header['creationdate'] = fid.read(16)

		header['datafilename'] = fid.read(64)

		header['orareserved'] = fid.read(256)

		if filesubtype == 11:
			# dyedag secific
			includesrefframe = fid.read(4) # 0 or 1
			header['includesrefframe'] = unpack(self.i,includesrefframe)[0]
			
			temp = fid.read(128)
			header['temp'] = temp
			header['listofstimuli'] = temp[0:np.max(np.where(temp!=0))+1]  # up to first non-zero stimulus
	
			ntrials = fid.read(4)
			header['ntrials'] = unpack(self.i,ntrials)[0]

			scalefactor = fid.read(4)
			header['scalefactor'] = unpack(self.i,scalefactor)[0] # bin * trials
	
			cameragain = fid.read(2)
			header['cameragain'] = unpack(self.h,cameragain)[0] # shcameragain        1,   2,   5,  10
	
			ampgain = fid.read(2)		    # amp gain            1,   4,  10,  16,
							    #                     40,  64, 100, 160,
							    #                     400,1000

			header['ampgain'] = unpack(self.h,ampgain)[0]

			samplingrate = fid.read(2)	    # sampling rate (1/x)
							    #                     1,   2,   4,   8,
							    #                     16,  32,  64, 128,
							    #                     256, 512,1024,2048
			header['samplingrate'] = unpack(self.h,samplingrate)[0]

			average = fid.read(2)		    # average             1,   2,   4,   8,
							    #                    16,  32,  64, 128
			header['average'] = unpack(self.h,average)[0]

			exposuretime = fid.read(2)	    # exposure time       1,   2,   4,   8,
							    #                     16,  32,  64, 128,
							    #                     256, 512,1024,2048
			header['exposuretime'] = unpack(self.h,exposuretime)[0]

			samplingaverage = fid.read(2)	    # sampling average    1,   2,   4,   8,
							    #                     16,  32,  64, 128
			header['samplingaverage'] = unpack(self.h,samplingaverage)[0]

			presentaverage = fid.read(2)
			header['presentaverage'] = unpack(self.h,presentaverage)[0]

			framesperstim = fid.read(2)
			header['framesperstim'] = unpack(self.h,framesperstim)[0]

			trialsperblock = fid.read(2)
			header['trialsperblock'] = unpack(self.h,trialsperblock)[0]

			sizeofanalogbufferinframes = fid.read(2)
			header['sizeofanalogbufferinframes'] = unpack(self.h,sizeofanalogbufferinframes)[0]

			cameratrials = fid.read(2)
			header['cameratrials'] = unpack(self.h,cameratrials)[0]

			header['filler'] = fid.read(106)

			header['dyedaqreserved'] = fid.read(256)

		else:
			# it's not dyedaq specific
			includesrefframe = fid.read(4)
			header['includesrefframe'] = unpack(self.i,includesrefframe)[0]

			header['listofstimuli'] = fid.read(256)

			nvideoframesperdataframe = fid.read(4)
			header['nvideoframesperdataframe'] = unpack(self.i,nvideoframesperdataframe)[0]

			ntrials = fid.read(4)
			header['ntrials'] = unpack(self.i,ntrials)[0]

			scalefactor = fid.read(4)
			header['scalefactor'] = unpack(self.i,scalefactor)[0]

			meanampgain = fid.read(4)
			header['meanampgain'] = unpack(self.i,meanampgain)[0]

			meanampdc = fid.read(4)
			header['meanampdc'] = unpack(self.i,meanampdc)[0]

			header['vdaqreserved'] = fid.read(256)

		header['user'] = fid.read(256)

		header['comment'] = fid.read(256)

		refscalefactor =fid.read(4)
		header['refscalefactor'] = unpack(self.i,refscalefactor)[0]
        		  # bin * trials for reference

		## End definitions of variables
		header['headersize']=fid.tell()
		fid.seek(0,2) # go to EOF
		header['actuallength'] = fid.tell() # report where EOF is in bytes

		fid.close()
		return header

	def get_data(self):
		"""Reads the data contained in the BLK file
		Returns
		-------
		data : numpy array
		    The data included in BLK file
		Raises
		------
		ValueError
		    If the headersize damaged
		RunTimeError
		    If data can't be read (the file is damaged)
		"""
		from struct import unpack
		#global_timer = datetime.datetime.now().replace(microsecond=0)
		filesize = self.header['filesize'] # Filesize extraction
		headersize = self.header['lenheader'] # Headersize extraction

		try:
			fid = open(self.filename,'rb') # Opening data file
		except:
			print('Cannot open file: ', self.filename) # If file can't be read
			return

		try:
			a=fid.seek(headersize,0) # go to data
		except ValueError:
			print('Cannot seek to byte: ', headersize)
			return

		try:
			data_in=fid.read() # Data recuperation
			format_string = str(int(self.H[0:-1])*(filesize-headersize)//2)+self.H[-1]
			data=unpack(format_string,data_in)

		except RuntimeError:
			print('Cannot read data')
			return

		fid.close()
		#print('Data extraction time: ',str(datetime.datetime.now().replace(microsecond=0)-global_timer))
		return data

	def get_time_frames_error(self):
		# Added by Isabelle Racicot on 25/08/2020. Problems with VDAQ saving data
		self.get_data()
		t_size=self.header['nframesperstim']
		filesize = self.header['filesize']
		framesize = self.header['framesize']
		headersize = self.header['headersize']
		t_size_header = int(round(float(filesize-headersize)/float(framesize),0))
		f = t_size_header/t_size
		return f

	def get_signal(self):
		"""Transformation of data linear bitstream to a regular image 2D + time data.
		It substitues old methods get_3d_image and get_4d_image.
		Parameters
		----------
		self object

		Returns
		-------
		image : numpy array
		    The 2D + time data image
		"""
		#global_timer = datetime.datetime.now().replace(microsecond=0)
		# Recuperation of image attributes
		x_size=self.header['framewidth']
		y_size=self.header['frameheight']
		z_size=1
		t_size=self.header['nframesperstim']
		filesize = self.header['filesize']
		framesize = self.header['framesize']
		headersize = self.header['headersize']

		# Check on time dimension aka number of frames
		t_size_header = int(round(float(filesize-headersize)/float(framesize),0))
		if  t_size_header!=t_size:
			f = t_size_header/t_size
			print('Number of time frames does not correspond to file size by a factor ',f)
			t_size = t_size_header
		a = self.data
		# Detrending
		#if detrend:
		#	a = np.reshape(a, (t_size, z_size*y_size*x_size))
		#	a =  signal.detrend(np.nan_to_num(a))# + np.mean(a, axis=0)
		print(len(a))
		a = np.reshape(a,(t_size,z_size,y_size,x_size)) # Transformation of data linear bitstream to a regular image 2D + time data
		a = np.reshape(a[:,0,:,:], (t_size, y_size,x_size))
		#print('vsdi-signal extraction time: ',str(datetime.datetime.now().replace(microsecond=0)-global_timer))
		return a
        

	def bin_signal(self):
		"""
		Binning image method. It partially reproduces the old get_3d_image and get_4d_image methods
		-for the temporal binning side- but introduce a resizing with interpolation for spatial binning.
		MATLAB interpolation is bicubic: in this version linear is used for fastness and performance showed.
		Parameters
		----------
		x_size: int 
			width of frame
		y_size: int
			height of frame
		temporal_binning : int
			Number of consecutive temporal samples to be averaged into the imported file
		spatial_binning : int
			Size (in pixels) of square window to be averaged into one pixel of the imported file
		vsdi_signal : BLKFile.image np.array (70, 1000, 1300)
		Returns
		-------
		image : numpy array
			The 2D + time data image resized
		"""
		#global_timer = datetime.datetime.now().replace(microsecond=0)
		x_size = self.header['framewidth']
		y_size = self.header['frameheight']
		t_size = self.header['nframesperstim']
		b = self.signal
		if self.temporal_binning > 1:
			t_size_binned = int( np.ceil( float(t_size) / self.temporal_binning ) )
			# print 't_size_binned if: ', t_size_binned
			# b = np.zeros( [ t_size_binned, a.shape[1], a.shape[2] ] )
			b = np.zeros((t_size_binned, y_size, x_size), dtype='float64')
			for t in range(t_size_binned):
				ind_min = t*self.temporal_binning
				ind_max = min((t+1)*self.temporal_binning, t_size )
				b[t,:,:] = self.image[ind_min:ind_max,:,:].mean(0).astype('float64')
		else:
			t_size_binned = t_size

		if self.spatial_binning > 1:
			x_bnnd_size = x_size//self.spatial_binning 
			y_bnnd_size = y_size//self.spatial_binning
			tmp = np.zeros((t_size_binned, y_bnnd_size, x_bnnd_size))
			for i in range(t_size_binned):
				tmp[i, :, :] = cv.resize(np.array(b[i, :, :], dtype='float64'), (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_LINEAR)
				#tmp[i, :, :] = cv.resize(b[i, :, :], (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_CUBIC)
			b = tmp
		#print('binning time: ',str(datetime.datetime.now().replace(microsecond=0)-global_timer))
		return b.astype('float64')


	def motion_index(self):
		"""
		TESTING: different behaviour from MATLAB. Interpolation returns different result.
		-------- 
		Motion index method: it computes an index used for automatically discarding the BLK on binned signal.
		It deletes and normalizes the BLK signal for the mean value of each frame. 
		Afterward the first frame is deleted and used for normalization.
		Parameters
		--------
		selfObject: numpy.array shape (self.header['nframesperstim'], self.header['framewidth'], self.header['frameheight'])
		bnnd_img
		Returns
		--------
		self.motion_ind : a float representing the motion index.
		self.motion_ind_max : the maximum value of motion index detected among the frames.
		"""
		#global_timer = datetime.datetime.now().replace(microsecond=0)
		#bnnd_img = self.bin_image()
		#bnnd_img = self.image #used for debugging 
		rshpd_img = np.transpose(np.reshape(self.binned_signal, (self.binned_signal.shape[0], self.binned_signal.shape[1]*self.binned_signal.shape[2])))# shape(15k,70)
		tmp = (rshpd_img-np.mean(rshpd_img, axis = 0))/(np.mean(rshpd_img, axis = 0))
		tmp_ = np.nan_to_num((tmp - tmp[:, 0].reshape(-1, 1 ))/tmp[:, 0].reshape(-1, 1))
		tmp_[:, 0] = tmp_[:, 1]
		mov = np.std(tmp_, axis=0)/np.sqrt(tmp_.shape[0])
		motion_ind = np.mean(mov) # 0.0170
		motion_ind_max = np.max(mov) # 0.0368
		print(motion_ind, motion_ind_max)
		#print('motion index computing time: ',str(datetime.datetime.now().replace(microsecond=0)-global_timer))
		return motion_ind, motion_ind_max

def circular_mask_roi(new_width, new_height):
	"""
	Creation of a circle of interest (ROI) mask. It uses the x bin and y bin of the BLK object.
	Parameter
	-----------
	self.header['framewidth']: int
	self.header['frameheight']: int
	self.spatial_binning: int
	Returns
	-----------
	roi_mask: numpy.array (width/spatial_binning, height/spatial_binning)
	Author: Kevin Blaize
	Matlab2Python conversion: Salvatore Giancani
	"""
	#global_timer = datetime.datetime.now().replace(microsecond=0)
	#new_width = (self.header['framewidth']//self.spatial_binning)
	#new_height = (self.header['frameheight']//self.spatial_binning)
	x = np.linspace(-1, 1, new_width)
	y = np.linspace(-1, 1, new_height)
	xv, yv = np.meshgrid(x, y)
	bnw = np.sqrt((np.square(xv)+np.square(yv))<0.5)
	mask=[~(bnw>0)]
	#print('mask roi building time: ',str(datetime.datetime.now().replace(microsecond=0)-global_timer))
	return mask # binned_x, binned_y

