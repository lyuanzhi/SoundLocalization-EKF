#include "pcmAnalysis.h"

PcmAnalysis::PcmAnalysis()
{
        this->numChannels = 10;
        this->sampleRate = 16000;
        this->bitsPerSample = 16;

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
                this->buffer[0][n] = this->buffer[3][n]; //(center point);
                // cout << endl;
        }
}

void PcmAnalysis::getInput(double *input, size_t size, double order)
{
        // 0.994127 0.988073 0.994127 1.00613 1.01207 1.00613
        // srand(time(0));
        // input[0] = 0.994127 + rand() % 1000 * 0.0001;
        // input[1] = 0.988073 + rand() % 1000 * 0.0001;
        // input[2] = 0.994127 + rand() % 1000 * 0.0001;
        // input[3] = 1.00613 + rand() % 1000 * 0.0001;
        // input[4] = 1.01207 + rand() % 1000 * 0.0001;
        // input[5] = 1.00613 + rand() % 1000 * 0.0001;
        // return;

        // 0.845899 1.02367 1.18946 1.09839 0.982255 0.899892
        // srand(time(0));
        // input[0] = 0.845899 + rand() % 1000 * 0.0001;
        // input[1] = 1.02367 + rand() % 1000 * 0.0001;
        // input[2] = 1.18946 + rand() % 1000 * 0.0001;
        // input[3] = 1.09839 + rand() % 1000 * 0.0001;
        // input[4] = 0.982255 + rand() % 1000 * 0.0001;
        // input[5] = 0.899892 + rand() % 1000 * 0.0001;
        // return;

        this->readData();

        // get buffer
        for(int n = 1; n <= size; n++){
                cout << n << ":" << endl;
                for(int i = 0; i < onceSampleNum; i++){
                        cout << this->buffer[n][i] << " ";
                }
                cout << endl << endl;
        }
        
        double A0 = 0;
        double Am = 0;
        double A0_max = 0;
        int A0_max_index = 0;
        double product = 1;
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

        for (int i = 5; i < onceSampleNum / 8 - 1; i++)
        {
                A0 = sqrt(out_0[i][0] * out_0[i][0] + out_0[i][1] * out_0[i][1]);
                if (A0_max < A0)
                {
                        A0_max = A0;
                        A0_max_index = i;
                }
        }
        // cout << A0_max_index << endl;

        for (int n = 1; n <= size; n++)
        {
                fftw_complex *in, *out;
                in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * onceSampleNum);
                for (int i = 0; i < onceSampleNum; i++)
                {
                        in[i][0] = this->buffer[n][i];
                        in[i][1] = 0;
                }
                this->p = fftw_plan_dft_1d(onceSampleNum, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_execute(this->p);
                fftw_destroy_plan(this->p);
                fftw_cleanup();

                Am = sqrt(out[A0_max_index][0] * out[A0_max_index][0] + out[A0_max_index][1] * out[A0_max_index][1]);
                input[n - 1] = pow(A0_max / Am, order);
                product *= input[n - 1];

                if (in != NULL)
                        fftw_free(in);
                if (out != NULL)
                        fftw_free(out);
        }
        for (int i = 0; i < size; i++)
        {
                // cout << input[i] <<" ";
                input[i] /= pow(product, 1.0 / (double)(size));
                // cout << input[i] <<" ";
        }
        // cout << endl << endl;
        if (in_0 != NULL)
                fftw_free(in_0);
        if (out_0 != NULL)
                fftw_free(out_0);
}

double PcmAnalysis::getDt()
{
        return (double)(onceSampleNum) / (double)(this->sampleRate);
}