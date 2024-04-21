#include <alsa/asoundlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <iomanip>
#include <stdio.h>
#include <fftw3.h>

#define onceSampleNum   480

using namespace std;

class PcmAnalysis
{
public:
        PcmAnalysis();
        ~PcmAnalysis();
        void getInput(double *, size_t, double);
        double getDt();

private:
        void readData();

private:
        snd_pcm_t *capture_handle;
        unsigned int numChannels;
        unsigned int sampleRate;
        uint16_t bitsPerSample;
        double buffer[7][onceSampleNum];
        fftw_plan p;
};