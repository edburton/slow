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
	float minChangeDrifted,maxChangeDrifted;
	float changeScalar,previousChangeScalar;
	static const float changeDrift=1/100.0f;
	float opacity;
	float Opacity;
	static const float opacityIntertia=1/10.0f;
	int camWidth, camHeight;
	int camY;
	static const int opacityN=6;
	float opacityMax;
	static const float diffScale=16;
	int diffWidth, diffHeight;
	bool firstFrame;
	qtime::MovieWriter	mMovieWriter;
	bool wroteFrame;
	bool debug;
	float time;
	float oldTimer;
	float previousChange;
	float totalDuration;
};

void ocvCaptureApp::setup()
{
	firstFrame=true;
	wroteFrame=false;
	minChange=maxChange=-1;
	opacityMax = 1/6.0f;
	oldTimer=changeScalar=totalDuration=previousChange=previousChangeScalar=opacity=0;
	mCapI=1;
	setFrameRate(400);
	int camWidth=640;
	int camH=camHeight=480;
	vector<Capture::DeviceRef> devices( Capture::getDevices() );
	for( vector<Capture::DeviceRef>::const_iterator deviceIt = devices.begin(); deviceIt != devices.end(); ++deviceIt ) {
		Capture::DeviceRef device = *deviceIt;
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
	debug=true;
	hideCursor();
	setFullScreen(true);
	mCapture=mCaptures[mCapI];
	mCapture.start();
}

void ocvCaptureApp::update()
{
	if( mCapture && mCapture.checkNewFrame() ) {
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
		float changeRangeDrifted=0;
		if (!firstFrame) {
			cv::absdiff(thisFrameBlurred,previousFrameBlurred,cvBlurredP3);
			cv::Scalar changes=cv::sum(cvBlurredP3);
			float change=sqrt((0.2126f*changes[2])+(0.7152*changes[1])+(0.0722*changes[0]));
			change=(change*opacityIntertia)+(previousChange*(1-opacityIntertia));
			previousChange=change;
			if (minChange<0)
				minChangeDrifted=maxChangeDrifted=minChange=maxChange=change;
			else {
				if (change<minChange)
					minChange=(change*opacityIntertia)+(minChange*(1-opacityIntertia));
				if (change>maxChange) 
					maxChange=(change*opacityIntertia)+(maxChange*(1-opacityIntertia));
				if (change<minChangeDrifted)
					minChangeDrifted=(change*opacityIntertia)+(minChangeDrifted*(1-opacityIntertia));
				if (change>maxChangeDrifted) 
					maxChangeDrifted=(change*opacityIntertia)+(maxChangeDrifted*(1-opacityIntertia));
			}
			changeRange=maxChange-minChange;
			changeRangeDrifted=maxChangeDrifted-minChangeDrifted;
			if (changeRangeDrifted>0 && (change-minChangeDrifted)>0)
				changeScalar=(change-minChangeDrifted)/changeRangeDrifted;
			else
				changeScalar=0;
		}
		thisFrameBlurred.copyTo(previousFrameBlurred);
		float drift=changeDrift*changeRangeDrifted;
		if (changeRangeDrifted>changeRange/10) {
			maxChangeDrifted-=drift;
			minChangeDrifted+=drift;	
		}
		changeScalar-=(1-changeScalar)*drift;
		changeScalar=(changeScalar*opacityIntertia)+(previousChangeScalar*(1-opacityIntertia));
		previousChangeScalar=changeScalar;
		if (changeScalar>1)
			changeScalar=1;
		else if (changeScalar<0)
			changeScalar=0;
		opacity=changeScalar*opacityMax;
				if (opacity>0);
		cv::accumulateWeighted(cvP2,cvOut,opacity);
		if (getElapsedFrames()>50 && mMovieWriter&& !firstFrame && changeScalar>0) {
			float duration=time-oldTimer;
			mMovieWriter.addFrame( fromOcv( cvOut ),changeScalar*duration) ;
			wroteFrame=true;
			totalDuration+=duration;
		}
		oldTimer=time;
		time=getElapsedSeconds();
		firstFrame=false;
	}
	
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
} 

void ocvCaptureApp::draw()
{
	gl::clear();
	gl::setViewport( getWindowBounds() );
	gl::color(Color(1.0f,1.0f,1.0f));
	gl::draw( fromOcv( cvOut ), getWindowBounds());
	if (getElapsedFrames()>1 && debug) {
		float width=diffWidth*10;
		gl::draw( fromOcv( thisFrameBlurred), Rectf(0,110,width,110+diffHeight*10 ));
		gl::color(Color(0.0f,0.0f,0.0f));
		gl::drawSolidRect(Rectf(0.0f,0.0f,width, 110));
		gl::color(Color(1,1-changeScalar,1-changeScalar));
		gl::drawSolidRect(Rectf(0.0f,0.0f,width*changeScalar, 110));
		gl::drawString ("changeScalar="+toString(changeScalar),  Vec2f(0,30));
		gl::drawString ("totalDuration="+toString(totalDuration),  Vec2f(0,50));
		gl::drawString ("Frame rate="+toString(getAverageFps()),  Vec2f(0,70));
		gl::drawString ("changeRangeDrifted="+toString(maxChangeDrifted-minChangeDrifted),  Vec2f(0,90));
		if (wroteFrame) {
			gl::drawString ("###RECORDING###",  Vec2f(0,10));
			wroteFrame=!wroteFrame;
		}
		
	}
}


CINDER_APP_BASIC( ocvCaptureApp, RendererGl )
