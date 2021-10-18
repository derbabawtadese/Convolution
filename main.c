/*
	1. This program filters high frequency signals above the filter cut-off frequency by convolving the input signal x[n] 
	with FIR filter kernel h[n] using several DSP computation methods:
		a. Direct form
		b. Convolution table
		c. Matrix form
		d. LTI form
	
	2. CPU clock source Freq: @168.00MHz using internal PLL
	3. uController - STM32F407VGTx

 Author:
    Derbabaw Tadese
 Date: 
    10/11/2021
*/

#include "stm32f4xx_hal.h"              // Keil::Device:STM32Cube HAL:Common 
#include "arm_math.h"                   // ARM::CMSIS:DSP
#include "math.h"
#include "input_signal.h"
#include "filter_coeff.h"

#define LY  		(SIG_LENGTH_1kHz_15kHz + (FIR_FILTER_LENGTH - 1))				/* length of conv. resualt */
#define LZH  		(SIG_LENGTH_1kHz_15kHz + 2 * (FIR_FILTER_LENGTH - 1))			/* length of zero padded input with offset both sides Lh - 1 were */
#define LZX  		(FIR_FILTER_LENGTH + 2 * (SIG_LENGTH_1kHz_15kHz - 1))			/* length of zero padded input with offset both sides Lx - 1 */
#define MIN(X,Y)	(((X) > (Y)) ? (Y) : (X))
#define MAX(X,Y)	(((X) > (Y)) ? (X) : (Y))

extern void SystemClock_Config(void);

/* plot functions */
void plotConvSignal(float32_t *y, int length, float32_t *data);
void plotSignal(float32_t *signal, int length, float32_t *data);

void arrReverse(float32_t *arr, int size);
void arrCopyAndPadd(float32_t *dest, float32_t *src, int offset, int signalLength);
void convXH(float32_t * h, int hLength, float32_t *x, float32_t *y);
void convHX(float32_t * h, float32_t *x, int xLength, float32_t *y);
void convHX_DirectMN( float32_t * h, int hLength, float32_t *x, int xLength, float32_t *y);
void convHX_MatrixForm( float32_t * h, int hLength, float32_t *x, int xLength, float32_t *y);

/* plot helper variables */
float32_t convxh, convhx, convhx_mn, convhx_matrix, conv_cmsis;
float32_t inputSignal, impResponse, freq;

/* buffers to store the signal which is shited and padded with zeros both sides */	
float32_t x[LZH];																										
float32_t h[LZX];															
float32_t y[SIG_LENGTH_1kHz_15kHz + (FIR_FILTER_LENGTH - 1)];																						/* output signal with length of nx + nh - 1 */

int main()
{
	HAL_Init();
	SystemClock_Config();
//	freq = HAL_RCC_GetHCLKFreq();
	
	plotSignal(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH, &impResponse);
	plotSignal(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, &inputSignal);
	
	arrCopyAndPadd(x, inputSignal_f32_1kHz_15kHz, FIR_FILTER_LENGTH, SIG_LENGTH_1kHz_15kHz);					/* padded x input with Lh - 1 zeros both sides */
	arrReverse(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH);								/* impulse signal reversing */
	
	/* computing y[n] = sum(k=0 -> Lh -1: h[n]*x[n-k]) */
	convXH(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH, x, y);
	plotConvSignal(y, LY, &convxh);
	
	arrCopyAndPadd(h, impResponseFIR_Coeff_f32_cf_4kHz, SIG_LENGTH_1kHz_15kHz, FIR_FILTER_LENGTH);					/* padded h impulse response with Lx - 1 zeros both sides */
	arrReverse(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz);									/* input signal signal reversing */
	
	/* computing y[n] = sum(k=0 -> Lx -1: x[n]*h[n-k]) */
	convHX(h, inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, y);
	
//	plotSignal(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, &inputSignal);
	plotConvSignal(y, LY, &convhx);
	
	arrReverse(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz);									/* input signal signal reversing */
	convHX_DirectMN(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH, 
	inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, y);
	
//	plotSignal(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, &inputSignal);
	plotConvSignal(y, LY, &convhx_mn);
	
	arrReverse(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH);																			/* impulse signal reversing */											
	convHX_MatrixForm(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH, 
	inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, y);
	
//	plotSignal(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, &inputSignal);
	plotConvSignal(y, LY, &convhx_matrix);
	
	// arm cmsis dsp convolution
	arm_conv_f32(impResponseFIR_Coeff_f32_cf_4kHz, FIR_FILTER_LENGTH, 
	inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, y);
	
//	plotSignal(inputSignal_f32_1kHz_15kHz, SIG_LENGTH_1kHz_15kHz, &inputSignal);
	plotConvSignal(y, LY, &conv_cmsis);
	
	while (1)
    {
		;
    }
}

/******************arrReverse*********************
 reversing the the signal either x[n] or h[n] for computing the convolution
 the reversing depend on the convolution formula h[n] * x[n - k] or h[n - k] * x[n]

 input: x[n] or h[n]
 output: x[-n] or h[-n] in reverse order
**************************************************/

void arrReverse(float32_t *arr, int size)
{
	float32_t tmp;
	
	for(int i = 0, j = size - 1; i < j; i++, j--)
	{
		tmp = arr[j];
		arr[j] = arr[i];
		arr[i] = tmp;
	}
}

/**************arrCopyAndPadd*********************
 copy the array and then shifting it by Lx or LY amount, 
 depend on the convolution formula
 h[n] * x[n - k] or h[n - k] * x[n]

 input: x[n] or h[n]
 output: x[n] or h[n] shifted signal by Lx or LY amount
**************************************************/

void arrCopyAndPadd(float32_t *dest, float32_t *src, int offset, int signalLength)
{
	for(int i = offset - 1, j = 0; j < signalLength ; i++, j++)
	{
		dest[i] = src[j];
	}
}

 /***************convHX***************************
 algorithm: y[n] = sum(k=0 -> Lh -1: h[n] * x[n - k])
 computing the convultion were h[n] reversed and x[n] has padded
 with (Lh - 1) zeros both sides 
  
 [0 ............. 0 | x[0] x[1] ...............x[m-1] | 0 ............. 0]
 ------------------------------------------------------------------------
 h[n] h[n-1]......... h[0] ..........................						h[-k]
 .... h[n] h[n-1].... h[1] h[0]......................						h[1-k]
 ..............................................h[n] h[n-1].........h[0] 	h[n-k] 
 [	Lh - 1	    |		Lx		      |		Lh - 1	 ]
 [		    | y[0] y[1]................................ y[Lx+Lh-1]  Yn=sum[m : x(k)h(n-k)]

 input: impulse response and input signal
 output: convolution result y[n]
**************************************************/

void convXH( float32_t * h, int hLength, float32_t *x, float32_t *y)
{
	for(int i = 0; i < LY; i++)
	{
		y[i] = 0;
		for(int j = 0; j < hLength; j++)
		{	
			y[i] += h[j] * x[i + j];				/* convolution resualt */
		}
	}
}

 /**********************convHX*********************
 algorithm: y[n] = sum(k=0 -> Lx -1: h[n - k] * x[n]
 computing the convultion were x[n] reversed and h[n] has padded
 with (Lx - 1) zeros both sides
 
 [0 ............. 0 | h[0] h[1] ................h[n-1] | 0 ............. 0]
 -------------------------------------------------------------------------
 x[n] x[n-1].........x[0] ..........................						x[-k]
 .... x[n] x[n-1]....x[1] x[0]......................						x[1-k] 
 ...............................................x[n] x[n-1].........x[0] 	x[n-k] 
 -------------------------------------------------------------------------
 [	Lx - 1		|	Lh			|	Lx - 1	  ]
 [			| y[0] y[1]................................ y[Lx+Lh-1 ] Yn=sum[m : h(k)x(n-k)]

 input: impulse response and input signal
 output: convolution result y[n]
**************************************************/

void convHX( float32_t * h, float32_t *x, int xLength, float32_t *y)
{
	for(int i = 0; i < LY; i++)
	{
		y[i] = 0;
		for(int j = 0; j < xLength; j++)
		{
			y[i] += h[i + j] * x[j];				/* convolution resualt */
		}
	}
}

 /**********************convHX_DirectMN*********************
 algorithm: y[n] = sum(k=0 -> Lx -1: h[n - k] * x[n]
 computing the convultion were x[n] reversed and h[n] has padded
 with (Lx - 1) zeros both sides
 
 [0 ............. 0 | h[0] h[1] ................h[n-1] | 0 ............. 0]
 ------------------------------------------------------------------------
 x[n] x[n-1].........x[0] ..........................						x[-k]
 .... x[n] x[n-1]....x[1] x[0]......................						x[1-k] 
 ...............................................x[n] x[n-1].........x[0] 	x[n-k] 
 ------------------------------------------------------------------------
 [	Lx - 1	    |		Lh		       |	Lx - 1	  ]
 [		    | y[0] y[1]................................ y[Lx+Lh-1 ]  Yn=sum[m : h(k)x(n-k)]

 input: impulse response and input signal
 output: convolution result y[n]
**************************************************/

void convHX_DirectMN( float32_t * h, int hLength, float32_t *x, int xLength, float32_t *y)
{
	for(int n = 0; n < LY; n++)
	{
		y[n] = 0;
		for(int m = MAX(0, n-xLength+1); m <= MIN(n, hLength); m++)
		{
			y[n] += h[m] * x[n-m];				/* convolution resualt */
		}
	}
}

 /**********************convHX_MatrixForm*********************
 matrix form algorithm: y[n] = sum(n = i + j; i -> xLength; j -> hLength : h[j] * x[i])

 x[0].........x[1] .............. x[n-1] 
-------------------------------------------------
h[0]	|  h[0]x[0]	  	h[0]x[1]	h[0]x[n-1]
h[1]	|  h[1]x[0]	  	h[1]x[1]	h[1]x[n-1]
.	|.................................................
.	|.................................................
.	|.................................................
h[n-1]	|  h[n-1]x[0]	       h[n-1]x[1]	h[n-1]x[n-1]
--------------------------------------------------

 input: impulse response and input signal
 output: convolution result y[n]
**************************************************/

void convHX_MatrixForm( float32_t * h, int hLength, float32_t *x, int xLength, float32_t *y)
{
	int i = 0, j = 0;
	for(; i < xLength; i++)
	{
		y[i + j] = 0;
		for(j = 0; j < hLength; j++)
		{
			y[i + j] += h[j] * x[i];				/* convolution resualt */
		}
	}
}

 /***************plotConvSignal*********************
 plot the filtered signal 
 input: filtered signal -> convolution result y[n]
 output: none
****************************************************/

void plotConvSignal(float32_t *y, int length, float32_t *data)
{
	for(int i = 0; i < length; i++)
	{
		*data = y[i];
		HAL_Delay(1);
		//if(i == length - 1) i = 0;
	}
}

 /*****************plotSignal*********************
 plot the unfiltered signal
 input: unfiltered signal
 output: none
**************************************************/

void plotSignal(float32_t *signal, int length, float32_t *data)
{
	for(int i = 0; i < length; i++)
	{
		*data = signal[i];
		HAL_Delay(1);
		//(i == length - 1) i = 0;
	}
}

/**************SysTick_Handler*********************
 dummy exception handler
 input: None
 output: none
**************************************************/

void SysTick_Handler(void)
{
	HAL_IncTick();
	HAL_SYSTICK_IRQHandler();
}
