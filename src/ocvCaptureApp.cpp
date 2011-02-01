#include "cinder/app/AppBasic.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/gl/Fbo.h"
#include "cinder/ip/grayscale.h"
#include "cinder/qtime/MovieWriter.h"
#include "cinder/Utilities.h"
#include "CinderOpenCV.h"
#include "cinder/Rand.h"
#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
using namespace std;
using namespace ci;
using namespace ci::app;
using namespace boost::posix_time;
using namespace boost::algorithm;

class ocvCaptureApp : public AppBasic {
public:
	void setup();
	void update();
	void keyDown( KeyEvent event );
	void draw();
	void updateLOG();
	
	vector<Capture>	mCaptures;
	Capture			mCapture;
	int				mCapI;
	cv::Mat			cvOut;
	cv::Mat			cvP1;
	cv::Mat			cvP2;
	cv::Mat			thisFrameBlurred;
	cv::Mat			cvBlurredP3;
	cv::Mat			previousFrameBlurred;
	float change;
	float minChange,maxChange;
	float minChangeDrifted,maxChangeDrifted;
	float changeScalar,previousChangeScalar;
	float opacity;
	int camFrameCount;
	static const float changeDrift=1/100.0f;
	static const float opacityIntertia=1/6.0f;
	static const int windowL=40;
	float windowA[windowL];
	float windowT;
	static const int cvL=20;
	vector<cv::Mat>	cvOutL;
	int cvNowI;
	int cvThenI;
	static const float maxOpacity=1/10.0f;
	int camWidth, camHeight;
	int camY;
	static const float diffScale=16;
	int diffWidth, diffHeight;
	bool firstFrame;
	qtime::MovieWriter	mMovieWriter;
	bool debug;
	bool showFramerate;
	float time;
	float oldTimer;
	int pleaseQuitCount;
	bool pleaseQuit;
	float duration;
	float totalDuration;
	float speed;
	int framesWritten;
	static const int LOGlength=640;
	static const int LOGs=6;
	int LOGi;
	float LOG[2][LOGs][LOGlength];
	string LOGlable[LOGs];
	float *LOGp[LOGs];
	Color LOGc[LOGs];
	bool wroteFrame;
	float t;
};

void ocvCaptureApp::setup()
{
	cvNowI=0;
	cvThenI=1;
	windowT=0;
	for (int i=0;i<windowL;i++) {
		float x=(i/(windowL-1.0f))-0.5f;
		x*=M_PI*2;
		x=(cos(x)+1)/2;
		windowT+=windowA[i]=x;
	}
	int c=0;
	LOGlable[c]="change";
	LOGp[c++]=&change;
	LOGlable[c]="minChange";
	LOGp[c++]=&minChange;
	LOGlable[c]="maxChange";
	LOGp[c++]=&maxChange;
	LOGlable[c]="minChangeDrifted";
	LOGp[c++]=&minChangeDrifted;
	LOGlable[c]="maxChangeDrifted";
	LOGp[c++]=&maxChangeDrifted;
	LOGlable[c]="changeScalar";
	LOGp[c++]=&changeScalar;
	for (int n=0;n<2;n++)
		for (int i=0;i<LOGlength;i++) 
			for (int s=0;s<LOGs;s++)
				LOG[n][s][i]=0;
	
	for (int s=0;s<LOGs;s++) {
		float h=s*(1.0f/LOGs);
		LOGc[s]=Color( CM_HSV, h, 1.0f, 1.0f );
	}
	firstFrame=true;
	wroteFrame=pleaseQuit=false;
	minChangeDrifted=maxChangeDrifted=minChange=maxChange=0.5f;
	changeScalar=1;
	previousChangeScalar=opacity=t=speed=camFrameCount=LOGi=mCapI=time=oldTimer=change=framesWritten=duration=totalDuration=pleaseQuitCount=opacity=0;
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
	mCapI=devices.size()-1;
	camHeight=(int)((1200.0f/1920)*camWidth);
	diffWidth=round(camWidth/diffScale);
	diffHeight=round(camHeight/diffScale);
	camY=(camH-camHeight)/2;
	string name=to_simple_string(second_clock::universal_time());
	string path = getDocumentsDirectory()+"slow["+name+"].mov";
	replace(path.begin(), path.end(), ':', '-');
	console() << path << std::endl;
	if( !path.empty() )
	{	
		qtime::MovieWriter::Format format;
		format.setCodec( qtime::MovieWriter::CODEC_MP4);
		format.setQuality( 0.9f );
		format.setDefaultDuration(1/10.0f);
		mMovieWriter = qtime::MovieWriter( path, camWidth, camHeight, format );
	}
	cvP1=cv::Mat(camHeight,camWidth,CV_32FC3);
	cvP2=cv::Mat(camHeight,camWidth,CV_32FC3);
	cvOut = cv::Mat(camHeight,camWidth,CV_32FC3);
	for (int n=0;n<cvL+1;n++)
		cvOutL.push_back(cv::Mat(camHeight,camWidth,CV_32FC3));
	thisFrameBlurred = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	previousFrameBlurred = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	cvBlurredP3 = cv::Mat(diffWidth,diffHeight,CV_32FC3);
	setWindowSize(camWidth, camHeight);
	showFramerate=debug=true;
	//hideCursor();
	//setFullScreen(true);
	mCapture=mCaptures[mCapI];
	mCapture.start();
}

void ocvCaptureApp::update()
{
	wroteFrame=false;
	//try {
	if(!pleaseQuit && camFrameCount>10 && mCapture && mCapture.checkNewFrame() ) {
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
		cv::resize(cvP2,cvBlurredP3,cv::Size(diffWidth,diffHeight),0,0,CV_INTER_AREA);
		cv::medianBlur( cvBlurredP3, cvBlurredP3, 5 );
		cv::GaussianBlur(cvBlurredP3, thisFrameBlurred , cv::Size(15,15), 0);
		
		if (++cvNowI>cvL)
			cvNowI=0;
		if (++cvThenI>cvL)
			cvThenI=0;
		cvP2.copyTo(cvOutL[cvNowI]);
		
		
		float changeRange=0;
		float changeRangeDrifted=0;
		
		if (!firstFrame) {
			cv::absdiff(thisFrameBlurred,previousFrameBlurred,cvBlurredP3);
			cv::Scalar changes=cv::sum(cvBlurredP3);
			change=sqrt((0.2126f*changes[2])+(0.7152*changes[1])+(0.0722*changes[0]));
			LOG[1][0][LOGi]=LOG[0][0][LOGi]=change;
			int i=LOGi;
			float t=0;
			for (int n=0;n<windowL;n++) {
				t+=LOG[0][0][i]*windowA[n];
				if (--i<0)
					i=LOGlength-1;
			}
			change=t/windowT;
			
			if (camFrameCount<400)
				change=0.5f*((400-camFrameCount)/400.0f)+change*(camFrameCount/400.0f);
			LOG[1][0][LOGi]=change;
			if (change>minChange)
				minChange=(change*(changeDrift/2))+(minChange*(1-(changeDrift/2)));
			else
				minChange=(change*(opacityIntertia))+(minChange*(1-(opacityIntertia)));
			if (change<maxChange) 
				maxChange=(change*(changeDrift/2))+(maxChange*(1-(changeDrift/2)));
			else
				maxChange=(change*(opacityIntertia))+(maxChange*(1-(opacityIntertia)));
			if (change<minChangeDrifted)
				minChangeDrifted=(change*opacityIntertia)+(minChangeDrifted*(1-opacityIntertia));
			if (change>maxChangeDrifted) 
				maxChangeDrifted=(change*opacityIntertia)+(maxChangeDrifted*(1-opacityIntertia));
			
			changeRange=maxChange-minChange;
			changeRangeDrifted=maxChangeDrifted-minChangeDrifted;
			if (changeRangeDrifted>0)
				changeScalar=(change-minChangeDrifted)/changeRangeDrifted;
			else
				changeScalar=0;
			
		}
		thisFrameBlurred.copyTo(previousFrameBlurred);
		
		firstFrame=false;
		if (camFrameCount>20) {
			
			float drift=changeDrift*changeRangeDrifted;
			if (changeRangeDrifted>changeRange/10)
				maxChangeDrifted-=drift;
			minChangeDrifted+=drift;	
			if (changeScalar>1)
				changeScalar=1;
			else if (changeScalar<0)
				changeScalar=0;
			changeScalar*=changeScalar;
			opacity=changeScalar*maxOpacity;
			if (opacity>0) {
				cv::accumulateWeighted(cvOutL[cvThenI],cvOut,opacity);
			}
			oldTimer=time;
			time=getElapsedSeconds();
			float realDuration=time-oldTimer;
			//console() << " realDuration=" << toString(realDuration*cvL) << " cvDrawF=" << toString(cvDrawF) << endl;
			float duration=oldTimer>0?realDuration:0;
			duration*=changeScalar;
			if (mMovieWriter && camFrameCount>30 && opacity>0 && duration>0) {
				ImageSourceRef image = fromOcv(cvOut);
				if (image) {	
					mMovieWriter.addFrame(image ,duration) ;
					
					wroteFrame=true;
					if (oldTimer>0)
						totalDuration+=duration;
					speed=duration/realDuration;
				}
				
			}
			else
				speed=0;
		}
		updateLOG();
	}
	//} catch (exception & ) {console() << "update ouch" << endl;}
	camFrameCount++;
	if (pleaseQuit)
		pleaseQuitCount++;
	if (mMovieWriter && pleaseQuitCount==2)
		try {
			mMovieWriter.finish();
		} catch( exception & ) {
			console() << "finish ouch" << endl;
		}
	if (pleaseQuitCount==1) {
		showCursor();
		setFullScreen(false);
	}
	if (pleaseQuitCount==3)
		AppBasic::quit();
}

void ocvCaptureApp::updateLOG() {
	LOGi=++LOGi%LOGlength;
	for (int s=1;s<LOGs;s++)
		LOG[1][s][LOGi]=*LOGp[s];
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
	else if(event.getChar() == 'q' )
		pleaseQuit=true;
} 

void ocvCaptureApp::draw()
{
	try {
		Area bounds=getWindowBounds() ;
		gl::setViewport( bounds  );
		gl::color(Color(1.0f,1.0f,1.0f));
		
		
		ImageSourceRef image = fromOcv(cvOut);
		if (image) {
			gl::draw( image, bounds);
		}
		
		
		if (showFramerate)
			gl::drawStringRight ("Frame rate="+toString(getAverageFps()),  Vec2f(bounds.getWidth()-5,bounds.getHeight()-20),Color(1.0f,0.0f,0.0f));
		if (debug) {
			gl::color(Color(1.0f,0.0f,0.0f));
			float gY=bounds.getHeight()/3;
			float gHeight=bounds.getHeight()/3;
			gl::drawLine(Vec2f(0,gY),Vec2f(bounds.getWidth(),gY));
			gl::drawLine(Vec2f(0,gY+gHeight),Vec2f(bounds.getWidth(),gY+gHeight));
			float sw=bounds.getWidth()/(LOGlength-1.0f);
			int i1=LOGi;
			int i2=i1>0?i1-1:LOGlength-1;
			for (int i=0;i<LOGlength-1;i++) {
				for (int s=0;s<LOGs;s++) {
					gl::color(LOGc[s]);
					float x=i*sw;
					float y1=(gY+gHeight)-(LOG[1][s][i1]*gHeight);
					float y2=(gY+gHeight)-(LOG[1][s][i2]*gHeight);
					gl::drawLine(Vec2f(x,y1),Vec2f(x+sw,y2));
				}
				if (--i1<0)
					i1=LOGlength-1;
				if (--i2<0)
					i2=LOGlength-1;
			}
			float ly=gY+5;
			for (int s=0;s<LOGs;s++) {
				gl::drawStringRight (toString(LOG[1][s][LOGi])+"<-"+LOGlable[s],  Vec2f(bounds.getWidth()-5,ly),LOGc[s]);
				ly+=20;
			}
			gl::drawString("Duration="+toString(totalDuration),  Vec2f(5,5),Color(1.0f,0.0f,0.0f));
			gl::drawString("Speed="+toString(speed),  Vec2f(5,25),Color(1.0f,0.0f,0.0f));
			if (wroteFrame)
				gl::drawString(">RECORDING!",  Vec2f(5,45),Color(1.0f,0.0f,0.0f));
		}
	} catch (exception & ) {console() << "draw ouch" << endl;}
}


CINDER_APP_BASIC( ocvCaptureApp, RendererGl )
