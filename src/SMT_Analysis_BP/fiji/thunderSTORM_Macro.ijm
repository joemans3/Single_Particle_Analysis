datapaths = newArray("/Volumes/Baljyot_HD/SMT_Olympus/20231017/rp_ez_fixed/Movie/rp_ez_fixed_2.tif",
"/Volumes/Baljyot_HD/SMT_Olympus/20231017/rp_ez_fixed/Movie/rp_ez_fixed_7.tif");
respaths = "/Volumes/Baljyot_HD/SMT_Olympus/20231017/rp_ez_fixed/Movie/TS_Analysis/"
uniquename = "rp_ez_fixed_"

counter = 1
for(i = 0; i < datapaths.length; i++) {
	open(datapaths[i]);
	run("Camera setup", "offset=100.0 isemgain=false photons2adu=0.23 pixelsize=130.0");
	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=100 estimator=[PSF: Gaussian] sigma=1.5 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=false nmax=3 fixed_intensity=true expected_intensity=500:2500 pvalue=1.0E-6 renderer=[Averaged shifted histograms] magnification=10.0 colorizez=false threed=false shifts=2 repaint=50");
	run("Show results table", "action=drift magnification=5.0 method=[Cross correlation] save=false steps=5 showcorrelations=true");
	run("Show results table","action=merge zcoordweight=0.1 offframes=2 dist=65.0 framespermolecule=0");
	run("Show results table","action=duplicates distformula=uncertainty");
	run("Export results", "filepath=["+respaths+uniquename+counter+".csv"+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true detections=true");
	close("*");
	counter= counter+1;
}