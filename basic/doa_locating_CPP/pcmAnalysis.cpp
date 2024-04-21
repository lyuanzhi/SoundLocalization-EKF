#include "pcmAnalysis.h"

PcmAnalysis::PcmAnalysis()
{
        this->numChannels = 10;
        this->sampleRate = 16000;
        this->bitsPerSample = 16;
        this->limit = ceil(20 + 0.06 / 340.0 * (double)(this->sampleRate));

        snd_pcm_hw_params_t *hw_params;

        if (snd_pcm_open(&this->capture_handle, "dsnooped_6_2", SND_PCM_STREAM_CAPTURE, 0))
        {
                cout << "Unable to open capture PCM device." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_malloc(&hw_params))
        {
                cout << "Cannot allocate hardware parameter structure." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_any(this->capture_handle, hw_params))
        {
                cout << "Cannot initialize hardware parameter structure." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_set_access(this->capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED))
        {
                cout << "Error setting interleaved mode." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_set_format(this->capture_handle, hw_params, SND_PCM_FORMAT_S16_LE))
        {
                cout << "Error setting format." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_set_channels(this->capture_handle, hw_params, this->numChannels))
        {
                cout << "Error setting channels." << endl;
                exit(-1);
        }

        if (snd_pcm_hw_params_set_rate_near(this->capture_handle, hw_params, &this->sampleRate, 0))
        {
                cout << "Error setting sampling rate." << endl;
                exit(-1);
        }

        /* Write the parameters to the driver */
        if (snd_pcm_hw_params(this->capture_handle, hw_params) < 0)
        {
                cout << "Unable to set HW parameters." << endl;
                exit(-1);
        }

        cout << "Open capture device is successful." << endl;
        if (hw_params)
                snd_pcm_hw_params_free(hw_params);

        if (snd_pcm_prepare(this->capture_handle) < 0)
        {
                cout << "无法使用音频接口." << endl;
                exit(-1);
        }
}

PcmAnalysis::~PcmAnalysis()
{
        fftw_destroy_plan(this->p);
        fftw_cleanup();
        snd_pcm_close(this->capture_handle);
}

void PcmAnalysis::readData()
{
        uint16_t data[onceSampleNum * this->numChannels];
        double double_data = 0;
        int err = 0;

        while (err != onceSampleNum)
        {
                // cout << "Reread." << endl;
                err = snd_pcm_readi(capture_handle, data, onceSampleNum);
                if (err == -EAGAIN || (err >= 0 && err < onceSampleNum))
                {
                        snd_pcm_wait(capture_handle, 1000);
                }
                else if (err == -EPIPE)
                {
                        snd_pcm_prepare(capture_handle);
                }

                if (err < 0)
                {
                        err = snd_pcm_recover(capture_handle, err, 0);
                        if (err < 0)
                        {
                                cout << "Read data err." << endl;
                                exit(-1);
                        }
                }
        }

        for (int n = 0; n < onceSampleNum; n++)
        {
                for (int i = 0; i < this->numChannels; i++)
                {
                        //原码转补码
                        if (data[n * this->numChannels + i] >= pow(2, this->bitsPerSample - 1))
                        {
                                double_data = data[n * this->numChannels + i] - pow(2, this->bitsPerSample);
                        }
                        else
                        {
                                double_data = data[n * this->numChannels + i];
                        }
                        if (i < 6)
                        {
                                this->buffer[i + 1][n] = double_data;
                        }
                        // cout << setw(10) << double_data;
                }
                this->buffer[0][n] = this->buffer[1][n]; //(center point);
                // cout << endl;
        }
}

void PcmAnalysis::getDelay_CC(double *delays, size_t size)
{
        this->readData();
        double sum = 0;
        for (int i = 1; i <= size; i++)
        {
                double R12 = 0;
                double R12_max = 0;
                int R12_max_index = 0;
                for (int m = -(limit - 1); m <= (limit - 1); m++)
                {
                        for (int n = limit * 2; n < onceSampleNum - limit * 2; n++)
                        {
                                R12 += this->buffer[0][n] * this->buffer[i][n + m];
                        }
                        if (R12_max < R12)
                        {
                                R12_max = R12;
                                R12_max_index = m;
                        }
                        R12 = 0;
                }
                delays[i - 1] = (double)(R12_max_index) / (double)(this->sampleRate);
                sum += delays[i - 1];
        }
        for(int i = 0; i < size; i++){
                delays[i] -= sum / (double)(size);
        }
}

void PcmAnalysis::getDelay_CPS(double *delays, size_t size)
{
        this->readData();
        double sum = 0;
        fftw_complex *in_0, *out_0;
        in_0 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
        out_0 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
        for (int i = 0; i < onceSampleNum; i++)
        {
                in_0[i][0] = this->buffer[0][i];
                in_0[i][1] = 0;
        }
        this->p = fftw_plan_dft_1d(onceSampleNum, in_0, out_0, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(this->p);
        fftw_destroy_plan(this->p);
        fftw_cleanup();

        for (int n = 1; n <= size; n++)
        {
                fftw_complex *in, *out, *inverse_in, *inverse_out;
                in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                inverse_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                inverse_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                double final_out[onceSampleNum];
                int max_index = 0;
                double max_value = 0;
                for (int i = 0; i < onceSampleNum; i++)
                {
                        in[i][0] = this->buffer[n][i];
                        in[i][1] = 0;
                }
                this->p = fftw_plan_dft_1d(onceSampleNum, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_execute(this->p);
                fftw_destroy_plan(this->p);
                fftw_cleanup();
                for (int i = 0; i < onceSampleNum; i++)
                {
                        inverse_in[i][0] = out[i][0] * out_0[i][0] + out[i][1] * out_0[i][1];
                        inverse_in[i][1] = out[i][1] * out_0[i][0] - out[i][0] * out_0[i][1];
                }
                this->p = fftw_plan_dft_1d(onceSampleNum, inverse_in, inverse_out, FFTW_BACKWARD, FFTW_ESTIMATE);
                fftw_execute(this->p);
                fftw_destroy_plan(this->p);
                fftw_cleanup();
                for (int i = 0; i < onceSampleNum; i++)
                {
                        if(i < limit || i > onceSampleNum - limit){
                                // [0~+lag_max，-lag_max~0)
                                final_out[i] = sqrt(inverse_out[i][0] * inverse_out[i][0] + inverse_out[i][1] * inverse_out[i][1]) / onceSampleNum / onceSampleNum;
                                if (final_out[i] > max_value)
                                {
                                        max_value = final_out[i];
                                        if (i < onceSampleNum / 2)
                                                max_index = i;
                                        else
                                                max_index = i - onceSampleNum;
                                }
                        }
                }
                // cout << "max_index: " << max_index << endl;
                delays[n - 1] = (double)(max_index) / (double)(this->sampleRate);
                sum += delays[n - 1];

                if (in != NULL)
                        fftw_free(in);
                if (out != NULL)
                        fftw_free(out);
                if (inverse_in != NULL)
                        fftw_free(inverse_in);
                if (inverse_out != NULL)
                        fftw_free(inverse_out);
        }
        for(int i = 0; i < size; i++){
                delays[i] -= sum / (double)(size);
        }

        if (in_0 != NULL)
                fftw_free(in_0);
        if (out_0 != NULL)
                fftw_free(out_0);
}

void PcmAnalysis::getDelay_GCC_PHAT(double *delays, size_t size)
{
        this->readData();
        // -6.10737e-05 -0.000124026 -6.10737e-05 6.36941e-05 0.000125523 6.36941e-05

        // delays[0] = -6.25e-05;
        // delays[1] = -0.000125;
        // delays[2] = -6.25e-05;
        // delays[3] = 6.25e-05;
        // delays[4] = 0.000125;
        // delays[5] = 6.25e-05;

        // delays[0] = -6.10737e-05;
        // delays[1] = -0.000124026;
        // delays[2] = -6.10737e-05;
        // delays[3] = 6.36941e-05;
        // delays[4] = 0.000125523;
        // delays[5] = 6.36941e-05;
        
        // return;

        fftw_complex *in_0, *out_0;
        in_0 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
        out_0 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
        double sum = 0;
        for (int i = 0; i < onceSampleNum; i++)
        {
                in_0[i][0] = this->buffer[0][i];
                in_0[i][1] = 0;
        }
        this->p = fftw_plan_dft_1d(onceSampleNum, in_0, out_0, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(this->p);
        fftw_destroy_plan(this->p);
        fftw_cleanup();

        for (int n = 1; n <= size; n++)
        {
                fftw_complex *in, *out, *inverse_in, *inverse_out;
                in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                inverse_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                inverse_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                double final_out[onceSampleNum];
                int max_index = 0;
                double max_value = 0;
                for (int i = 0; i < onceSampleNum; i++)
                {
                        in[i][0] = this->buffer[n][i];
                        in[i][1] = 0;
                }
                this->p = fftw_plan_dft_1d(onceSampleNum, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_execute(this->p);
                fftw_destroy_plan(this->p);
                fftw_cleanup();
                double H = 0;
                for (int i = 0; i < onceSampleNum; i++)
                {
                        inverse_in[i][0] = out[i][0] * out_0[i][0] + out[i][1] * out_0[i][1];
                        inverse_in[i][1] = out[i][1] * out_0[i][0] - out[i][0] * out_0[i][1];
                        H = sqrt(inverse_in[i][0] * inverse_in[i][0] + inverse_in[i][1] * inverse_in[i][1]);
                        inverse_in[i][0] /= H;
                        inverse_in[i][1] /= H;
                }
                this->p = fftw_plan_dft_1d(onceSampleNum, inverse_in, inverse_out, FFTW_BACKWARD, FFTW_ESTIMATE);
                fftw_execute(this->p);
                fftw_destroy_plan(this->p);
                fftw_cleanup();
                for (int i = 0; i < onceSampleNum; i++)
                {
                        if(i < limit || i > onceSampleNum - limit){
                                // [0~+lag_max，-lag_max~0)
                                final_out[i] = sqrt(inverse_out[i][0] * inverse_out[i][0] + inverse_out[i][1] * inverse_out[i][1]) / onceSampleNum / onceSampleNum;
                                if (final_out[i] > max_value)
                                {
                                        max_value = final_out[i];
                                        if (i < onceSampleNum / 2)
                                                max_index = i;
                                        else
                                                max_index = i - onceSampleNum;
                                }
                        }
                }
                // cout << "m: " << max_index << " ";
                delays[n - 1] = (double)(max_index) / (double)(this->sampleRate);
                sum += delays[n - 1];
                if (in != NULL)
                        fftw_free(in);
                if (out != NULL)
                        fftw_free(out);
                if (inverse_in != NULL)
                        fftw_free(inverse_in);
                if (inverse_out != NULL)
                        fftw_free(inverse_out);
        }
        for(int i = 0; i < size; i++){
                delays[i] -= sum / (double)(size);
        }
        if (in_0 != NULL)
                fftw_free(in_0);
        if (out_0 != NULL)
                fftw_free(out_0);
}

double PcmAnalysis::getDt()
{
        return (double)(onceSampleNum) / (double)(this->sampleRate);
}