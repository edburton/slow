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
	
	vector<Capture>	mCaptures;
	Capture			mCapture;
	int				mCapI;
	cv::Mat			cvP1;
	cv::Mat			cvP2;
	cv::Mat			cvOut;
	cv::Mat			cvBlurredP3;
	vector<cv::Mat>	cvInput;
	vector<cv::Mat>	cvBlurredThumbnails;
	cv::Mat cvLastSmoothedThumbnail;
	float change;
	float changeGrounded;
	float changeThresholdMin;
	float changeThresholdMargin;
	float changeSoftRange;
	float changeRangeK;
	float changeFieldK;
	float changeLift;
	float changeFriction;
	float absoluteMaxChange;
	float minChange,maxChange;
	float minChangeV,maxChangeV;
	float changeScalar;
	float opacity;
	int camFrameCount;
	static const int smoothLength=15;
	float windowA[smoothLength];
	int smoothPeakPosition;
	int camDelay;
	int cvOutputNow;
	int cvOutputThen;
	int cvBlurredThumbnailNow;
	static const float maxOpacity=1.0f/6;
	int camWidth, camHeight;
	int camY;
	static const float diffScale=8;
	int diffWidth, diffHeight;
	bool firstFrame;
	qtime::MovieWriter	mMovieWriter;
	bool debug;
	bool showFramerate;
	bool showValues;
	float time;
	float oldTimer;
	int pleaseQuitCount;
	bool pleaseQuit;
	float duration;
	float totalDuration;
	float speed;
	float maxSpeed;
	int framesWritten;
	static const int LOGlength=640;
	static const int LOGs=5;
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
	maxSpeed=1;
	changeThresholdMin=100;
	changeThresholdMargin=0.1f;
	changeSoftRange=0.0f;
	changeRangeK=1/12.0f;
	changeLift=1/8.0f;
	changeFieldK=2;
	changeFriction=0.1f;
	cvBlurredThumbnailNow=cvOutputNow=0;
	cvOutputThen=1;
	float maxT=-1;
	float total=0;
	for (int i=0;i<smoothLength;i++) {
		float x=(i+0.5f)/(float)(smoothLength);
		x=sin(x*M_PI);
		x*=x;
		windowA[i]=x;
		total+=x;
		if (x>maxT) {
			smoothPeakPosition=i;
			maxT=x;
		}
	}
	camDelay=(int)round(smoothPeakPosition*2.5f); //should be 3? not sure why 2.5 works better
	//console() << "smoothLength=" << toString(smoothLength)<< " smoothPeakPosition=" << toString(smoothPeakPosition) << endl;
	for (int i=0;i<smoothLength;i++)
		windowA[i]/=total;
	int c=0;
	LOGlable[c]="change";
	LOGp[c++]=&change;
	LOGlable[c]="changeScalar";
	LOGp[c++]=&changeScalar;
	LOGlable[c]="changeGrounded";
	LOGp[c++]=&changeGrounded;
	LOGlable[c]="minChange";
	LOGp[c++]=&minChange;
	LOGlable[c]="maxChange";
	LOGp[c++]=&maxChange;
	for (int n=0;n<2;n++)
		for (int i=0;i<LOGlength;i++) 
			for (int s=0;s<LOGs;s++)
				LOG[n][s][i]=0;//0 used for change smoothing, 1 for changeScalar smoothing 
	
	for (int s=0;s<LOGs;s++) {
		float h=(s-1)*(1.0f/(LOGs-1));
		if (h<0)
			h=0;
		LOGc[s]=Color( CM_HSV, h, 1.0f, 1.0f );
	}
	firstFrame=true;
	wroteFrame=pleaseQuit=false;
	minChange=maxChange=0.5f;
	changeScalar=1;
	absoluteMaxChange=changeGrounded=minChangeV=maxChangeV=opacity=t=speed=camFrameCount=LOGi=mCapI=time=oldTimer=change=framesWritten=duration=totalDuration=pleaseQuitCount=opacity=0;
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
	cvP1=cv::Mat(camHeight,camWidth,CV_32FC3,cv::Scalar(0,0,0));
	cvP2=cv::Mat(camHeight,camWidth,CV_32FC3,cv::Scalar(0,0,0));
	for (int n=0;n<camDelay;n++)
		cvInput.push_back(cv::Mat(camHeight,camWidth,CV_32FC3,cv::Scalar(0,0,0)));
	cvOut = cv::Mat(camHeight,camWidth,CV_32FC3,cv::Scalar(0,0,0));
	cvBlurredP3 = cv::Mat(diffHeight,diffWidth,CV_32FC3,cv::Scalar(0,0,0));
	for (int n=0;n<smoothLength+1;n++)
		cvBlurredThumbnails.push_back(cv::Mat(diffHeight,diffWidth,CV_32FC3,cv::Scalar(0,0,0)));
	cvLastSmoothedThumbnail = cv::Mat(diffHeight,diffWidth,CV_32FC3,cv::Scalar(0,0,0));
	setWindowSize(camWidth, camHeight);
	showFramerate=false;
	debug=false;
	showValues=false;
	hideCursor();
	setFullScreen(true);
	mCapture=mCaptures[mCapI];
	mCapture.start();
}

void ocvCaptureApp::update()
{
	wroteFrame=false;
	try {
		if(!pleaseQuit && mCapture && mCapture.checkNewFrame() ) {
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
			cvOutputNow=++cvOutputNow%camDelay;
			cvOutputThen=++cvOutputThen%camDelay;
			cvBlurredThumbnailNow=++cvBlurredThumbnailNow%(smoothLength+1);
			cv::flip(cvP1,cvInput[cvOutputNow],1);
			cv::resize(cvInput[cvOutputNow],cvBlurredP3,cv::Size(diffWidth,diffHeight),0,0,CV_INTER_AREA);
			cv::GaussianBlur(cvBlurredP3, cvBlurredThumbnails[cvBlurredThumbnailNow] , cv::Size(15,15), 0);
			float changeRange=0;
			float previousChange=changeGrounded;
			int indx=LOGi;
			if (!firstFrame) {
				int index=cvBlurredThumbnailNow;
				cv::Mat cvThisSmoothedThumbnail = cv::Mat(diffHeight,diffWidth,CV_32FC3,cv::Scalar(0,0,0));
				for (int i=0;i<smoothLength;i++) {
					cv::accumulateWeighted(cvBlurredThumbnails[index],cvThisSmoothedThumbnail,windowA[i]);
					if (--index<0)
						index=smoothLength;
				}
				cv::absdiff(cvThisSmoothedThumbnail,cvLastSmoothedThumbnail,cvBlurredP3);
				cv::Scalar changes=cv::sum(cvBlurredP3);
				change=sqrt((0.2126f*changes[2])+(0.7152*changes[1])+(0.0722*changes[0]));
				cvThisSmoothedThumbnail.copyTo(cvLastSmoothedThumbnail);
				if (change>absoluteMaxChange)
					absoluteMaxChange=change;
				
				changeThresholdMin=-1;
				for (int i=0;i<LOGlength;i++) {
					if (--indx<0)
						indx=LOGlength-1;
					if ((changeThresholdMin<0 || LOG[1][0][indx]<changeThresholdMin) && LOG[1][0][indx]>0)
						changeThresholdMin= LOG[1][0][indx];			
				}
				LOG[0][0][LOGi]=change;
				int i=LOGi;
				float t=0;
				int c=0;
				for (int n=0;n<smoothLength;n++) {
					float f=LOG[0][0][i];
					if (f>0) {
						t+=f*windowA[n];
						c++;
					}
					if (--i<0)
						i=LOGlength-1;
				}
				if (c==smoothLength)
					LOG[1][0][LOGi]=t;
				//console() << "LOG[1][0][LOGi]=" << toString(LOG[1][0][LOGi]) << endl;
				changeGrounded=t-(changeThresholdMin+changeThresholdMargin);
				
				float d=(maxChange-minChange)-changeSoftRange;
				minChangeV+=changeLift;
				minChangeV+=d*changeRangeK;
				maxChangeV-=d*changeRangeK;
				minChangeV*=changeFriction;
				maxChangeV*=changeFriction;
				minChange+=minChangeV;
				maxChange+=maxChangeV;
				if (changeGrounded<minChange) {
					minChange=changeGrounded;
				}
				if (changeGrounded>maxChange) {
					maxChange=changeGrounded;
				}
				float changeChange=changeGrounded-previousChange;
				float absChangeChange=abs(changeChange);
				if (absChangeChange>0) {
					absChangeChange=sqrt(absChangeChange);
					float dMin=((changeGrounded)-minChange)*80;
					if (dMin<0)
						dMin=0;
					float fMin=(absChangeChange*changeFieldK)*sin(M_PI/(dMin+2));
					minChangeV-=fMin;
					float dMax=(maxChange-(changeGrounded))*80;
					if (dMax<0)
						dMax=0;
					float fMax=0.5f*(absChangeChange*changeFieldK)*sin(M_PI/(dMax+2));
					maxChangeV+=fMax;
				}
				float minChangeZeroed=minChange<0?0:minChange;
				changeRange=maxChange-minChangeZeroed;
				if (changeRange>0 && changeGrounded>0)
					changeScalar=(changeGrounded-minChangeZeroed)/changeRange;
				else
					changeScalar=0;
			}
			
			firstFrame=false;
			if (changeScalar>1)
				changeScalar=1;
			else if (changeScalar<0)
				changeScalar=0;
			//dchangeScalar*=changeScalar;
			
			LOG[0][1][LOGi]=changeScalar;
			int ix=LOGi;
			t=0;
			for (int n=0;n<smoothLength;n++) {
				t+=LOG[0][1][ix]*windowA[n];
				if (--ix<0)
					ix=LOGlength-1;
			}
			changeScalar=LOG[1][1][LOGi]=t;
			
			
			opacity=changeScalar*maxOpacity;
			if (opacity>0)
				cv::accumulateWeighted(cvInput[cvOutputThen],cvOut,opacity);
			oldTimer=time;
			time=getElapsedSeconds();
			float realDuration=oldTimer>0?time-oldTimer:0;
			float duration=maxSpeed*realDuration;
			duration*=changeScalar;
			if (mMovieWriter && opacity>0 && duration>0) {
				totalDuration+=duration;
				speed=duration/realDuration;
				ImageSourceRef image = fromOcv(cvOut);
				if (image) {
					mMovieWriter.addFrame(image ,duration) ;
					wroteFrame=true;
				}
			}
			else
				speed=0;
			for (int s=2;s<LOGs;s++)
				LOG[0][s][LOGi]=LOG[1][s][LOGi]=*LOGp[s];
			LOGi=++LOGi%LOGlength;
		}
	} catch (exception & ) {console() << "update ouch" << endl;}
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
	//if (camFrameCount%100==0)
	//	console() << toString(getAverageFps())+"<-Frame rate" << endl;
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
	else if(event.getChar() == 'v' )
		showValues=!showValues;
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
		if (showFramerate) {
			gl::drawStringRight (toString(getAverageFps())+"<-Frame rate",  Vec2f(bounds.getWidth()-5,5),Color(0.25f,0.0f,0.0f));
			gl::drawStringRight(toString(totalDuration)+"<-Duration",  Vec2f(bounds.getWidth()-5,25),Color(0.25f,0.0f,0.0f));
			if (wroteFrame) {
				string rs="RECORDING";
				float r=(speed/maxSpeed);
				float l=r*(bounds.getWidth()/2);
				gl::drawStringRight("RECORDING<",  Vec2f(bounds.getWidth()-(5+l),45),Color(0.25f,0.0f,0.0f));
				gl::color(Color(0.5f-(r/2),0.0f,0.0f));
				gl::drawSolidRect(Rectf(bounds.getWidth()-(5+l),45,bounds.getWidth()-5,55));
				gl::color(Color(1.0f,1.0f,1.0f));
			}
		}
		if (debug) {
			ImageSourceRef image = fromOcv(cvInput[cvOutputNow]);
			if (image) {
				gl::draw( image, Rectf(0,0,bounds.getWidth()/4,bounds.getHeight()/4));
			}
			image = fromOcv(cvLastSmoothedThumbnail);
			if (image) {
				gl::draw( image, Rectf(0,bounds.getHeight()/4,bounds.getWidth()/4,bounds.getHeight()/2));
			}
			image = fromOcv(cvInput[cvOutputThen]);
			if (image) {
				gl::draw( image, Rectf(bounds.getWidth()/4,bounds.getHeight()/4,bounds.getWidth()/2,bounds.getHeight()/2));
			}
			
			gl::color(Color(1.0f,0.0f,0.0f));
			float gY=bounds.getHeight()/2;
			float gHeight=bounds.getHeight()/4;
			gl::drawLine(Vec2f(0,gY),Vec2f(bounds.getWidth(),gY));
			gl::drawLine(Vec2f(0,gY+gHeight),Vec2f(bounds.getWidth(),gY+gHeight));
			//float thresh=(gY+gHeight)-(changeThresholdMin*gHeight);
			//gl::color(Color(0.5f,0.0f,0.0f));
			//gl::drawLine(Vec2f(0,thresh),Vec2f(bounds.getWidth(),thresh));
			//thresh=(gY+gHeight)-((changeThresholdMin-changeThresholdMargin)*gHeight);
			//gl::drawLine(Vec2f(0,thresh),Vec2f(bounds.getWidth(),thresh));
			//gl::color(Color(1.0f,0.0f,0.0f));
			float sw=bounds.getWidth()/(LOGlength-1.0f);
			int i=LOGi-1>0?LOGi-1:LOGlength-1;
			int i1=i;
			int i2=i1>0?i1-1:LOGlength-1;
			for (int i=0;i<LOGlength-1;i++) {
				for (int s=1;s<LOGs;s++) {
					gl::color(LOGc[s]);
					float x=i*sw;
					float y1=(gY+gHeight)-(LOG[1][s][i1]*gHeight);
					float y2=(gY+gHeight)-(LOG[1][s][i2]*gHeight);
					gl::drawLine(Vec2f(x,y1),Vec2f(x+sw,y2));
					if (s==1) {
						gl::drawLine(Vec2f(x+1,y1),Vec2f(x+sw+1,y2));
						gl::drawLine(Vec2f(x+1,y1+1),Vec2f(x+sw+1,y2+1));
						gl::drawLine(Vec2f(x,y1+1),Vec2f(x+sw,y2+1));
					}
				}
				if (--i1<0)
					i1=LOGlength-1;
				if (--i2<0)
					i2=LOGlength-1;
			}
		}
		if (showValues) {
			int i=LOGi-1>0?LOGi-1:LOGlength-1;
			float ly=5+bounds.getHeight()/2;
			for (int s=1;s<LOGs;s++) {
				gl::drawStringRight (toString(LOG[1][s][i])+"<-"+LOGlable[s],  Vec2f(bounds.getWidth()-5,ly),LOGc[s]);
				ly+=20;
			}
		}
	} catch (exception & ) {console() << "draw ouch" << endl;}
}

CINDER_APP_BASIC( ocvCaptureApp, RendererGl )
