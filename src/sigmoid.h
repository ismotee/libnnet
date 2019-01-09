#pragma once
#include <math.h>

/**********************************************************************/
/*** sigmoid.c:  This code contains the function routine            ***/
/***             sigmoid() which performs the unipolar sigmoid      ***/
/***             function for backpropagation neural computation.   ***/
/***             Accepts the input value x then returns it's        ***/
/***             sigmoid value in float.                            ***/
/***                                                                ***/
/***  function usage:                                               ***/
/***       float sigmoid(float x);                                  ***/
/***           x:  Input value                                      ***/
/***                                                                ***/
/***  Written by:  Kiyoshi Kawaguchi                                ***/
/***               Electrical and Computer Engineering              ***/
/***               University of Texas at El Paso                   ***/
/***  Last update:  09/28/99  for version 2.0 of BP-XOR program     ***/

/**********************************************************************/

float sigmoid(float x) {
    float exp_value;
    float return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double) -x);

    /*** Final sigmoid value ***/
    return_value = (float) 1 / (1 + exp_value);

    return return_value;
}
