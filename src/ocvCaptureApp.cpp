#include "cinder/app/AppBasic.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/gl/Fbo.h"
#include "cinder/ip/grayscale.h"
#include "CinderOpenCV.h"
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
	static const int FBO_WIDTH = 640, FBO_HEIGHT = 480;
	int opacityI;
	static const int opacityN=6;
	float opacitys[opacityN];
};

void ocvCaptureApp::setup()
{
	opacityI=mCapI=1;
	opacitys[0]= 1/200.0f;
	opacitys[1]= 1/100.0f;
	opacitys[2]= 1/20.0f;
	opacitys[3]= 1/10.0f;
	opacitys[4]= 1/5.0f;
	opacitys[5]= 1/1.0f;
	
	vector<Capture::DeviceRef> devices( Capture::getDevices() );
	for( vector<Capture::DeviceRef>::const_iterator deviceIt = devices.begin(); deviceIt != devices.end(); ++deviceIt ) {
		Capture::DeviceRef device = *deviceIt;
		console() << "Found Device " << device->getName() << " ID: " << device->getUniqueId() << std::endl;
		try {
			if( device->checkAvailable() ) {
				mCaptures.push_back( Capture( FBO_WIDTH, FBO_HEIGHT, device ) );
			}
			else
				console() << "device is NOT available" << std::endl;
		}
		catch( CaptureExc & ) {
			console() << "Unable to initialize device: " << device->getName() << endl;
		}	
	}	
	cvP1=cv::Mat(FBO_HEIGHT,FBO_WIDTH,CV_32FC3);
	cvP2=cv::Mat(FBO_HEIGHT,FBO_WIDTH,CV_32FC3);
	cvOut = cv::Mat(FBO_HEIGHT,FBO_WIDTH,CV_32FC3);
	setWindowSize(FBO_WIDTH, FBO_HEIGHT);
	//hideCursor();
	//setFullScreen(true);
	mCapture=mCaptures[mCapI];
	mCapture.start();
}

void ocvCaptureApp::update()
{
	if( mCapture && mCapture.checkNewFrame() ) {
		Surface surface=mCapture.getSurface();
		cv::Mat rgb=toOcv( surface ,CV_8UC3);
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
		cv::medianBlur( cvP2, cvP1, 5 );
		cv::accumulateWeighted(cvP1,cvOut,opacitys[opacityI]);
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
	} else if (event.getChar() == 's') {
		opacityI=++opacityI%opacityN;
	}
} 

void ocvCaptureApp::draw()
{
	gl::clear();
	gl::setViewport( getWindowBounds() );
	gl::color(Color(1.0f,1.0f,1.0f));
	gl::draw( fromOcv( cvOut ), getWindowBounds());
}


CINDER_APP_BASIC( ocvCaptureApp, RendererGl )
