#include "cinder/app/AppBasic.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/gl/Fbo.h"
#include "cinder/ip/grayscale.h"
#include "cinder/qtime/MovieWriter.h"
#include "cinder/Utilities.h"
#include "CinderOpenCV.h"
#include "cinder/Rand.h"
#include <vector>
#include <list>
using namespace std;
using namespace ci;
using namespace ci::app;

class ocvCaptureApp : public AppBasic {
public:
	void setup();
	void update();
	void keyDown( KeyEvent event );
	void draw();
	
	vector<Capture>	mCaptures;
	Capture			mCapture;
	int				mCapI;
	cv::Mat			cvOut;
	cv::Mat			cvP1;
	cv::Mat			cvP2;
	cv::Mat			cvP3;
	cv::Mat			thisFrameBlurred;
	cv::Mat			cvBlurredP3;
	cv::Mat			previousFrameBlurred;
	float minChange,maxChange;
	float change,changeDrifted;
	float minChangeDrifted,maxChangeDrifted;
	float changeScalar,previousChangeScalar;
	float changeScalarClamped;
	static const float changeDrift=1/10.0f;
	static const float maxOpacity=1/1.0f;
	float opacity;
	float opacityIntertia;
	int camWidth, camHeight;
	int camY;
	static const int opacityN=6;
	float opacityMax;
	static const float diffScale=16;
	const static float slowMow=16;
	int diffWidth, diffHeight;
	bool firstFrame;
	qtime::MovieWriter mMovieWriter;
	bool wroteFrame;
	bool debug;
	bool showFramerate;
	float time;
	float oldTimer;
	float previousChange;
	float totalDuration;
	int camFrameCount;
	static const int LOGlimit=640;
	int LOGCurrent;
	int LOGcount;
	float LOGcountF;
	float LOGchangeScalarClamped[LOGlimit];
	int lLimit;
	int pleaseQuitCount;
	bool pleaseQuit;
};

void ocvCaptureApp::setup()
{
	lLimit=LOGlimit;
	firstFrame=true;
	pleaseQuit=wroteFrame=false;
	minChange=maxChange=-1;
	opacityIntertia=1/10.0f;
	LOGcountF=pleaseQuitCount=camFrameCount=LOGcount=LOGCurrent=change=changeDrifted=oldTimer=changeScalar=totalDuration=previousChange=previousChangeScalar=opacity=0;
	mCapI=1;
	setFrameRate(400);
	int camWidth=640;
	int camH=camHeight=480;
	vector<Capture::DeviceRef> devices( Capture::getDevices() );
	for( vector<Capture::DeviceRef>::const_iterator deviceIt = devices.begin(); deviceIt != devices.end(); ++deviceIt ) {
		Capture::DeviceRef device =*deviceIt;
		console() << "Found Device " << device->getName() << " ID: " << device->getUniqueId() << std::endl;
		try {
			if( device->checkAvailable() ) {
				mCaptures.push_back( Capture( camWidth, camHeight, device ) );
			}
			else
				console() << "device is NOT available" << std::endl;
		}
		catch( CaptureExc & ) {
			console() << "Unable to initialize device: " << device->getName() << endl;
		}	
	}
	camHeight=(int)((1200.0f/1920)*camWidth);
	diffWidth=round(camWidth/diffScale);
	diffHeight=round(camHeight/diffScale);
	camY=(camH-camHeight)/2;
	Rand::randomize();
	string path = getDocumentsDirectory()+"slow"+toString(Rand::randInt())+".mov";
	if( !path.empty() )
	{	
		qtime::MovieWriter::Format format;
		format.setCodec( qtime::MovieWriter::CODEC_MP4);
		format.setQuality( 0.9f );
		format.setDefaultDuration(1/20.0f);
		format.enableReordering(false);
		mMovieWriter = qtime::MovieWriter( path, camWidth, camHeight, format );
	}
	cvP1=cv::Mat(camHeight,camWidth,CV_32FC3);
	cvP2=cv::Mat(camHeight,camWidth,CV_32FC3);
	cvP3=cv::Mat(camHeight,camWidth,CV_32FC3);
	cvOut = cv::Mat(camHeight,camWidth,CV_32FC3);
	thisFrameBlurred = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	previousFrameBlurred = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	cvBlurredP3 = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	setWindowSize(camWidth, camHeight);
	showFramerate=debug=true;
	hideCursor();
	setFullScreen(true);
	mCapture=mCaptures[mCapI];
	mCapture.start();
}

void ocvCaptureApp::update()
{
	if(camFrameCount>1 && mCapture && mCapture.checkNewFrame() ) {
		bool quickStart=totalDuration<200;
		Surface surface=mCapture.getSurface();
		cv::Mat rgb=toOcv( surface ,CV_8UC3);
		rgb.adjustROI(-camY, -camY, 0,0);
		vector<cv::Mat> planes;
		split(rgb, planes);
		cv::Scalar  sums=cv::mean(rgb);
		float max=0;
		for (int i=0;i<3;i++)
			if (sums[i]>max)
				max=sums[i];
		float v[3];
		for (int i=0;i<3;i++)
			v[i]=(sums[i]/max);
		//console() << "sums[" << v[0] << "," << v[1] << "," << v[2] << "]" << std::endl;
		cv::Mat blue(planes[0]),b;
		cv::equalizeHist(blue, b);
		cv::Mat green(planes[1]),g;
		cv::equalizeHist(green, g);
		cv::Mat red(planes[2]),r;
		cv::equalizeHist(red, r);
		r.convertTo(red, CV_32FC1,v[2]/255.0f);
		g.convertTo(green, CV_32FC1,v[1]/255.0f);
		b.convertTo(blue, CV_32FC1,v[0]/255.0f);
		int from_to[] = { 0,0,  1,1,  2,2 };
		cv::Mat bgr[] = { blue,green	,red};
		cv::mixChannels(bgr,3,&cvP1,1,from_to,3);
		cv::flip(cvP1,cvP2,1);
		//cv::medianBlur( cvP2, cvP1, 5 );
		cvP2.copyTo(cvP3);
		cv::resize(cvP2,cvBlurredP3,cv::Size(diffWidth,diffHeight),0,0,CV_INTER_AREA);
		cv::medianBlur( cvBlurredP3, cvBlurredP3, 5 );
		cv::GaussianBlur(cvBlurredP3, thisFrameBlurred , cv::Size(15,15), 0);
		float changeRange=0;
		float drift=0;
		float changeRangeDrifted;
		if (!firstFrame) {
			cv::absdiff(thisFrameBlurred,previousFrameBlurred,cvBlurredP3);
			cv::Scalar changes=cv::sum(cvBlurredP3);
			change=sqrt((0.2126f*changes[2])+(0.7152*changes[1])+(0.0722*changes[0]));
			changeDrifted=quickStart?change:(change*opacityIntertia)+(previousChange*(1-opacityIntertia));
			previousChange=change;
			if (minChange<0) {
				minChangeDrifted=maxChangeDrifted=minChange=maxChange=changeDrifted;
			}
			else {
				if (changeDrifted<minChange)
					minChange=quickStart?changeDrifted:(changeDrifted*opacityIntertia)+(minChange*(1-opacityIntertia));
				if (changeDrifted>maxChange) 
					maxChange=quickStart?changeDrifted:(changeDrifted*opacityIntertia)+(maxChange*(1-opacityIntertia));
				if (changeDrifted<minChangeDrifted)
					minChangeDrifted=quickStart?changeDrifted:(changeDrifted*opacityIntertia)+(minChangeDrifted*(1-opacityIntertia));
				if (changeDrifted>maxChangeDrifted) 
					maxChangeDrifted=quickStart?changeDrifted:(changeDrifted*opacityIntertia)+(maxChangeDrifted*(1-opacityIntertia));
			}
			changeRange=maxChange-minChange;
			changeRange=1/(maxChange-minChange);
			
		}
		thisFrameBlurred.copyTo(previousFrameBlurred);
		drift=changeDrift*changeRangeDrifted;
		float oldmaxChangeDrifted=maxChangeDrifted;
		if (maxChangeDrifted>minChangeDrifted+maxChange/10) 
			maxChangeDrifted-=drift;
		if (changeRangeDrifted<oldmaxChangeDrifted-maxChange/10) 
			minChangeDrifted+=drift;
		changeRangeDrifted=maxChangeDrifted-minChangeDrifted;
		if (changeRangeDrifted>0 && (changeDrifted-minChangeDrifted)>0)
			changeScalar=(changeDrifted-minChangeDrifted)/changeRangeDrifted;
		else
			changeScalar=0;
		changeScalar=(changeScalar*opacityIntertia)+(previousChangeScalar*(1-opacityIntertia));
		previousChangeScalar=changeScalar;
		changeScalar=pow(changeScalar,2);
		changeScalarClamped=changeScalar-0.5f;
		changeScalarClamped*=1+1/100.0f;
		changeScalarClamped+=0.5;
		if (changeScalarClamped>1)
			changeScalarClamped=1;
		else if (changeScalarClamped<0)
			changeScalarClamped=0;
		float duration=time-oldTimer;
		opacity=maxOpacity*changeScalar;
		duration*=slowMow*changeScalarClamped;
		float doubleFrameRate=2/getFrameRate();
		if (duration>2/getFrameRate())
			duration=doubleFrameRate;
		if (!pleaseQuit && mMovieWriter && camFrameCount>10 && opacity>0) {
			cv::accumulateWeighted(cvP2,cvOut,opacity);
			if (duration>0) {
				mMovieWriter.addFrame( fromOcv( cvOut ),duration) ;
				wroteFrame=true;
				LOGcountF+=min(changeScalarClamped*10,1.0f);
				LOGcount=floor(LOGcountF);
				LOGCurrent=LOGcount%lLimit;
				LOGchangeScalarClamped[LOGCurrent]=changeScalarClamped;
				totalDuration+=duration;
			}
		}
		oldTimer=time;
		time=getElapsedSeconds();
		firstFrame=false;
	}
	camFrameCount++;
	if (pleaseQuit)
		pleaseQuitCount++;
	if (pleaseQuitCount>2)
		mMovieWriter.finish();
	if (pleaseQuitCount>4)
		AppBasic::quit();
}


void ocvCaptureApp::keyDown( KeyEvent event )
{
	if( event.getChar() == 'f' ) {
		if (isFullScreen()) {
			showCursor();
			setFullScreen(false);
		} else{
			hideCursor();
			setFullScreen(true);
		}
	} else if(mCaptures.size()>1 && event.getChar() == 'c' ) {
		if (mCapture)
			mCapture.stop();
		mCapI=++mCapI%mCaptures.size();
		mCapture=mCaptures[mCapI];
		mCapture.start();
	} else if(event.getChar() == 'd' )
		debug=!debug;
	else if(event.getChar() == 'r' )
		showFramerate=!showFramerate;
	else if(mMovieWriter && event.getChar() == 'q' ) {
		pleaseQuit=true;
	}
} 

void ocvCaptureApp::draw()
{
	gl::clear();
	gl::setViewport( getWindowBounds() );
	gl::color(Color(1.0f,1.0f,1.0f));
	gl::draw( fromOcv( cvOut ), getWindowBounds());
	if (showFramerate)
		gl::drawString ("Frame rate="+toString(getAverageFps()),  Vec2f(5,100));
	if (debug) {
		float h=150;
		int n=5;
		gl::color(Color(1.0f,1.0f,1.0f));
		gl::draw( fromOcv( thisFrameBlurred), Rectf(0,h,diffWidth*10,h+diffHeight*10 ));
		gl::drawString ("totalDuration="+toString(totalDuration),  Vec2f(5,40));
		gl::drawString ("changeScalarClamped="+toString(changeScalarClamped),  Vec2f(5,70));
		gl::drawString ("show debug = 'd' show framerate = 'r' toggle fullscreen= 'f' toggle camera 'c' quit = 'q'",  Vec2f(5,130));
		if (wroteFrame) {
			wroteFrame=false;
			gl::drawString ("RECORDING!",  Vec2f(5,10));
			gl::color(Color(0.0f,0.0f,0.0f));
			for (int i=0;i<=n;i++) {
				float y=(i/(float)n)*h;
				gl::drawLine(Vec2f(0,y),Vec2f(getWindowWidth(),y));
			}
			gl::color(Color(1.0f,0.0f,0.0f));
			if (changeScalarClamped>0) {
				for (int i=0;i<=n;i++) {
					float y=(i/(float)n)*h;
					gl::drawLine(Vec2f(0,y),Vec2f(getWindowWidth()*changeScalarClamped,y));
				}
				gl::drawLine(Vec2f(getWindowWidth()*changeScalarClamped,0),Vec2f(getWindowWidth()*changeScalarClamped,h));
			}
		}
		gl::color(Color(1.0f,0.0f,0.0f));
		float ys=h;
		float xs=getWindowWidth()/(float)lLimit;
		for (int i=0;i<min(LOGcount,lLimit-1);i++) {
			int iL1=((lLimit+LOGcount)-i)%lLimit;
			int iL2=((lLimit+LOGcount)-(i+1))%lLimit;
			if (i>0&&iL1>0&&iL2>0) {
				float LOG1changeScalarClamped=LOGchangeScalarClamped[iL1];
				float LOG2changeScalarClamped=LOGchangeScalarClamped[iL2];
				float x1=i*xs;
				float x2=(i+1)*xs;
				gl::drawLine(Vec2f(x1,h*LOG1changeScalarClamped),Vec2f(x2,h*LOG2changeScalarClamped));
			}
		}						   
	}
	glEnd();
}


CINDER_APP_BASIC( ocvCaptureApp, RendererGl )
