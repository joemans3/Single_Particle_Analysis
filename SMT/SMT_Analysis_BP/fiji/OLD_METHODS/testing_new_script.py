import sys
import os
import os.path
import glob
from ij import IJ, ImagePlus, ImageStack
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.formats import ImageReader
from loci.formats import MetadataTools
from ome.units import UNITS
#from loci.formats.in import ND2Reader
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.detection import DogDetectorFactory
from fiji.plugin.trackmate.detection import DetectorKeys
from fiji.plugin.trackmate import Spot
from fiji.plugin.trackmate import SpotCollection
from fiji.plugin.trackmate.features import SpotFeatureCalculator
from fiji.plugin.trackmate.features.spot import SpotContrastAndSNRAnalyzerFactory
import sys
import csv
import os
# Get currently selected image
#imp = WindowManager.getCurrentImage()
dir_tiff = '/Volumes/Baljyot_HD/SMT_Olympus/Baljyot_temp/12/rpoc_m9'


#for (root, dirs, files) in os.walk(dir_tiff):
#    print(files)
#    files = [ fi for fi in files if fi.endswith("_seg.tif") ]
#    num_files = len(files)
files = glob.glob(dir_tiff + '/' +'*seg.tif')
root = dir_tiff
save_analysis_dir = dir_tiff + '/' + 'Analysis_test'


if not os.path.exists(save_analysis_dir):
	os.mkdir(save_analysis_dir)
print(files)
for z in range(len(files)):
	str_name_fil = files[z]
	print(str_name_fil)
    
	imp = IJ.openImage(str_name_fil)
    
	imp.show()
	
	dims = imp.getDimensions();
	imp.setDimensions( dims[ 2 ], dims[ 4 ], dims[ 3 ] );
	dims = imp.getDimensions();
	model = Model()
   

	model.setLogger(Logger.IJ_LOGGER)
	logger = Logger.IJ_LOGGER
	settings = Settings(imp)
	#settings.setFrom(imp)
      

	settings.detectorFactory = DogDetectorFactory()
	settings.detectorSettings = {
    	DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    	DetectorKeys.KEY_RADIUS : 2.0,
    	DetectorKeys.KEY_TARGET_CHANNEL : 1,
    	DetectorKeys.KEY_THRESHOLD : 1.0,
	
    	DetectorKeys.KEY_DO_MEDIAN_FILTERING : False,
	} 
    

	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 10.
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']= 80.
	settings.trackerSettings['MAX_FRAME_GAP']= 0

	settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
	settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
	settings.addSpotAnalyzerFactory(SpotRadiusEstimatorFactory())
	settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
   
	settings.initialSpotFilterValue = 1
   
	print(str(settings))

   
	trackmate = TrackMate(model, settings)

 
	ok = trackmate.checkInput()
	if not ok:
		print(str(trackmate.getErrorMessage()))
		sys.exit(str(trackmate.getErrorMessage()))
     
	ok = trackmate.process()
	if not ok:
		print(str(trackmate.getErrorMessage()))
		sys.exit(str(trackmate.getErrorMessage()))

      
      

	model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
    
	selectionModel = SelectionModel(model)
	displayer =  HyperStackDisplayer(model, selectionModel, imp)
	displayer.render()
	displayer.refresh()


	go = 0
	if go:
		sys.exit()
	print("hi")
	fm = model.getFeatureModel()

	a = model.getSpots().iterator(True)
	cd = root

	print(model.getTrackModel().trackIDs(True))
	for id in model.getTrackModel().trackIDs(True):
   
	    # Fetch the track feature from the feature model.
	    v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
	    model.getLogger().log('')
	    model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
	       
	    track = model.getTrackModel().trackSpots(id)
	    for spot in track:
	        sid = spot.ID()
	        # Fetch spot features directly from spot. 
	        x=spot.getFeature('POSITION_X')
	        y=spot.getFeature('POSITION_Y')
	        t=spot.getFeature('FRAME')
	        rad=spot.getFeature('RADIUS')
	        q=spot.getFeature('QUALITY')
	        snr=spot.getFeature('SNR') 
	        mean=spot.getFeature('MEAN_INTENSITY')
	        model.getLogger().log('\tspot ID = ' + str(sid) + ': x='+str(x)+', y='+str(y)+', t='+str(t)+', q='+str(q) + ', snr='+str(snr) + ', mean = ' + str(mean))
	        dir_name_save_inten = save_analysis_dir + '/' + files[z][:-4] + '.tif_spots' +'.csv'
	        with open(dir_name_save_inten, 'a') as f:
	        	spamWriter = csv.writer(f, delimiter=',')
	        	spamWriter.writerow([str(id),t,x,y,mean])






