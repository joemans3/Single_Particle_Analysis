from ij import IJ, ImagePlus, ImageStack
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.SelectionModel as SelectionModel
import fiji.plugin.trackmate.TrackMate as TrackMate
import fiji.plugin.trackmate.Logger as Logger
import fiji.plugin.trackmate.detection.DetectorKeys as DetectorKeys
import fiji.plugin.trackmate.detection.DogDetectorFactory as DogDetectorFactory
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.FeatureAnalyzer as FeatureAnalyzer
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory as SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.action.ExportStatsToIJAction as ExportStatsToIJAction
import fiji.plugin.trackmate.io.TmXmlReader as TmXmlReader
import fiji.plugin.trackmate.action.ExportTracksToXML as ExportTracksToXML
import fiji.plugin.trackmate.io.TmXmlWriter as TmXmlWriter
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.features.SpotFeatureCalculator as SpotFeatureCalculator
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzer as SpotContrastAndSNRAnalyzer
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer as TrackSpeedStatisticsAnalyzer
import fiji.plugin.trackmate.util.TMUtils as TMUtils
import fiji.plugin.trackmate.features.spot.SpotRadiusEstimatorFactory as SpotRadiusEstimatorFactory
import sys
import csv
import os
# Get currently selected image
#imp = WindowManager.getCurrentImage()
dir_tiff = '/Users/baljyot/Desktop/sd/Desktop/Baljyot_EXP_RPOC/Scripts/0.001-0.01-0.1_100-100-100_r-r_10-10-10_1000_SD_0.5_test'


#for (root, dirs, files) in os.walk(dir_tiff):
#    files = [ fi for fi in files if fi.endswith(".tif") ]
#    num_files = len(files)

files = ["0.001-0.01-0.1_100-100-100_r-r_10-10-10_1000_SD_0.5_test_1_seg.tif"]
root = dir_tiff
save_analysis_dir = dir_tiff + '/' + 'Analysis'


if not os.path.exists(save_analysis_dir):
	os.mkdir(save_analysis_dir)
print(files)
for z in range(len(files)):
	str_name_fil = root + '/' + files[z]
	print(str_name_fil)
    
	imp = IJ.openImage(str_name_fil)
    
	imp.show()
	
	dims = imp.getDimensions();
	imp.setDimensions( dims[ 2 ], dims[ 4 ], dims[ 3 ] );
	dims = imp.getDimensions();
	model = Model()
   

	model.setLogger(Logger.IJ_LOGGER)

	settings = Settings()
	settings.setFrom(imp)
      

	settings.detectorFactory = DogDetectorFactory()
	settings.detectorSettings = {
    	DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    	DetectorKeys.KEY_RADIUS : 4.0,
    	DetectorKeys.KEY_TARGET_CHANNEL : 1,
    	DetectorKeys.KEY_THRESHOLD : 1.0,	
    	DetectorKeys.KEY_DO_MEDIAN_FILTERING : False,
	} 
    

	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 100.
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
	cd = "/Users/baljyot/Desktop/Baljyot_EXP_RPOC/Scripts"

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




