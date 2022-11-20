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
dir_tiff = '/Users/baljyot/Documents/2019-2020/RNAP_PAPER/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/segmented'


for (root, dirs, files) in os.walk(dir_tiff):
    files = [ fi for fi in files if fi.endswith("seg.tif") ]
    num_files = len(files)

    
save_analysis_dir = dir_tiff + '/' + 'Analysis'


if not os.path.exists(save_analysis_dir):
	os.mkdir(save_analysis_dir)
print(files)
for z in range(len(files)):
	str_name_fil = root + '/' + files [z]
	print(str_name_fil)
    
	imp = IJ.openImage(str_name_fil)
    
	imp.show()
   
   
	model = Model()
   

	model.setLogger(Logger.IJ_LOGGER)

	settings = Settings()
	settings.setFrom(imp)
      

	settings.detectorFactory = DogDetectorFactory()
	settings.detectorSettings = {
    	DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    	DetectorKeys.KEY_RADIUS : 2.5,
    	DetectorKeys.KEY_TARGET_CHANNEL : 1,
    	DetectorKeys.KEY_THRESHOLD : 5.,
    	DetectorKeys.KEY_DO_MEDIAN_FILTERING : True,
	} 
    

	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 5.
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']=8.
	settings.trackerSettings['MAX_FRAME_GAP']= 1

	settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
	settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
	settings.addSpotAnalyzerFactory(SpotRadiusEstimatorFactory())
	settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
   
	#settings.initialSpotFilterValue = 1
   
	print(str(settings))

   
	trackmate = TrackMate(model, settings)

 
	ok = trackmate.checkInput()
	if not ok:
		print(str(trackmate.getErrorMessage()))
		#sys.exit(str(trackmate.getErrorMessage()))
     
	ok = trackmate.process()
	if not ok:
		print(str(trackmate.getErrorMessage()))
		#sys.exit(str(trackmate.getErrorMessage()))

      
      

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
	cd = "D:\\Baljyot_Experiments\\20190524\\rpoc\\segmented"

	for i in a:
		x=i.getFeature("POSITION_X")
		y=i.getFeature("POSITION_Y")
		std=i.getFeature("STANDARD_DEVIATION")
		rad=i.getFeature("RADIUS")
		ed=i.getFeature("ESTIMATED_DIAMETER")
		dir_name_save_inten = save_analysis_dir + '/' + files[z] + '_' + 'spots' +'.csv'
		with open(dir_name_save_inten, 'a') as f:
			spamWriter = csv.writer(f, delimiter=',')
			spamWriter.writerow([x,y,rad,ed,std])