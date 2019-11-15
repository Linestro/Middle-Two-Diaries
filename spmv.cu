#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <fstream>
#include <time.h>

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (descr)              cusparseDestroyMatDescr(descr);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)

int main(){
    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int *    cooRowIndexHostPtr=0;
    int *    cooColIndexHostPtr=0;
    double * cooValHostPtr=0;
    int *    cooRowIndex=0;
    int *    cooColIndex=0;
    double * cooVal=0;
    int *    xIndHostPtr=0;
    double * xValHostPtr=0;
    double * yHostPtr=0;
    double * y_static=0;
    int *    xInd=0;
    double * xVal=0;
    double * y=0;
    int *    csrRowPtr=0;
    double * zHostPtr=0;
    double * z=0;
    int      n, nnz;
    double dzero =0.0;
    double done = 1.0;

    printf("testing example\n");
    /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0     |
       |    4.0             |
       |5.0     6.0 7.0     |
       |    8.0     9.0     |
       |                10.0| */

    n = 200;      // rank of the matrix
    nnz = 796;   // number of non-zero elements
    
    cooRowIndexHostPtr = (int *)   malloc(nnz*sizeof(cooRowIndexHostPtr[0]));
    cooColIndexHostPtr = (int *)   malloc(nnz*sizeof(cooColIndexHostPtr[0]));
    cooValHostPtr      = (double *)malloc(nnz*sizeof(cooValHostPtr[0]));
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)){
        CLEANUP("Host malloc failed (matrix)");
        return 1;
    }

    cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=-615.6962723589505;
    cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=1; cooValHostPtr[1]=310.0731361794753;
    cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=100; cooValHostPtr[2]=4.0;
    cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=0; cooValHostPtr[3]=310.0731361794753;
    cooRowIndexHostPtr[4]=1; cooColIndexHostPtr[4]=1; cooValHostPtr[4]=-615.6962723589505;
    cooRowIndexHostPtr[5]=1; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=310.0731361794753;
    cooRowIndexHostPtr[6]=1; cooColIndexHostPtr[6]=101; cooValHostPtr[6]=4.0;
    cooRowIndexHostPtr[7]=2; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=310.0731361794753;
    cooRowIndexHostPtr[8]=2; cooColIndexHostPtr[8]=2; cooValHostPtr[8]=-615.6962723589505;
    cooRowIndexHostPtr[9]=2; cooColIndexHostPtr[9]=3; cooValHostPtr[9]=310.0731361794753;
    cooRowIndexHostPtr[10]=2; cooColIndexHostPtr[10]=102; cooValHostPtr[10]=4.0;
    cooRowIndexHostPtr[11]=3; cooColIndexHostPtr[11]=2; cooValHostPtr[11]=310.0731361794753;
    cooRowIndexHostPtr[12]=3; cooColIndexHostPtr[12]=3; cooValHostPtr[12]=-615.6962723589505;
    cooRowIndexHostPtr[13]=3; cooColIndexHostPtr[13]=4; cooValHostPtr[13]=310.0731361794753;
    cooRowIndexHostPtr[14]=3; cooColIndexHostPtr[14]=103; cooValHostPtr[14]=4.0;
    cooRowIndexHostPtr[15]=4; cooColIndexHostPtr[15]=3; cooValHostPtr[15]=310.0731361794753;
    cooRowIndexHostPtr[16]=4; cooColIndexHostPtr[16]=4; cooValHostPtr[16]=-615.6962723589505;
    cooRowIndexHostPtr[17]=4; cooColIndexHostPtr[17]=5; cooValHostPtr[17]=310.0731361794753;
    cooRowIndexHostPtr[18]=4; cooColIndexHostPtr[18]=104; cooValHostPtr[18]=4.0;
    cooRowIndexHostPtr[19]=5; cooColIndexHostPtr[19]=4; cooValHostPtr[19]=310.0731361794753;
    cooRowIndexHostPtr[20]=5; cooColIndexHostPtr[20]=5; cooValHostPtr[20]=-615.6962723589505;
    cooRowIndexHostPtr[21]=5; cooColIndexHostPtr[21]=6; cooValHostPtr[21]=310.0731361794753;
    cooRowIndexHostPtr[22]=5; cooColIndexHostPtr[22]=105; cooValHostPtr[22]=4.0;
    cooRowIndexHostPtr[23]=6; cooColIndexHostPtr[23]=5; cooValHostPtr[23]=310.0731361794753;
    cooRowIndexHostPtr[24]=6; cooColIndexHostPtr[24]=6; cooValHostPtr[24]=-615.6962723589505;
    cooRowIndexHostPtr[25]=6; cooColIndexHostPtr[25]=7; cooValHostPtr[25]=310.0731361794753;
    cooRowIndexHostPtr[26]=6; cooColIndexHostPtr[26]=106; cooValHostPtr[26]=4.0;
    cooRowIndexHostPtr[27]=7; cooColIndexHostPtr[27]=6; cooValHostPtr[27]=310.0731361794753;
    cooRowIndexHostPtr[28]=7; cooColIndexHostPtr[28]=7; cooValHostPtr[28]=-615.6962723589505;
    cooRowIndexHostPtr[29]=7; cooColIndexHostPtr[29]=8; cooValHostPtr[29]=310.0731361794753;
    cooRowIndexHostPtr[30]=7; cooColIndexHostPtr[30]=107; cooValHostPtr[30]=4.0;
    cooRowIndexHostPtr[31]=8; cooColIndexHostPtr[31]=7; cooValHostPtr[31]=310.0731361794753;
    cooRowIndexHostPtr[32]=8; cooColIndexHostPtr[32]=8; cooValHostPtr[32]=-615.6962723589505;
    cooRowIndexHostPtr[33]=8; cooColIndexHostPtr[33]=9; cooValHostPtr[33]=310.0731361794753;
    cooRowIndexHostPtr[34]=8; cooColIndexHostPtr[34]=108; cooValHostPtr[34]=4.0;
    cooRowIndexHostPtr[35]=9; cooColIndexHostPtr[35]=8; cooValHostPtr[35]=310.0731361794753;
    cooRowIndexHostPtr[36]=9; cooColIndexHostPtr[36]=9; cooValHostPtr[36]=-615.6962723589505;
    cooRowIndexHostPtr[37]=9; cooColIndexHostPtr[37]=10; cooValHostPtr[37]=310.0731361794753;
    cooRowIndexHostPtr[38]=9; cooColIndexHostPtr[38]=109; cooValHostPtr[38]=4.0;
    cooRowIndexHostPtr[39]=10; cooColIndexHostPtr[39]=9; cooValHostPtr[39]=310.0731361794753;
    cooRowIndexHostPtr[40]=10; cooColIndexHostPtr[40]=10; cooValHostPtr[40]=-615.6962723589505;
    cooRowIndexHostPtr[41]=10; cooColIndexHostPtr[41]=11; cooValHostPtr[41]=310.0731361794753;
    cooRowIndexHostPtr[42]=10; cooColIndexHostPtr[42]=110; cooValHostPtr[42]=4.0;
    cooRowIndexHostPtr[43]=11; cooColIndexHostPtr[43]=10; cooValHostPtr[43]=310.0731361794753;
    cooRowIndexHostPtr[44]=11; cooColIndexHostPtr[44]=11; cooValHostPtr[44]=-615.6962723589505;
    cooRowIndexHostPtr[45]=11; cooColIndexHostPtr[45]=12; cooValHostPtr[45]=310.0731361794753;
    cooRowIndexHostPtr[46]=11; cooColIndexHostPtr[46]=111; cooValHostPtr[46]=4.0;
    cooRowIndexHostPtr[47]=12; cooColIndexHostPtr[47]=11; cooValHostPtr[47]=310.0731361794753;
    cooRowIndexHostPtr[48]=12; cooColIndexHostPtr[48]=12; cooValHostPtr[48]=-615.6962723589505;
    cooRowIndexHostPtr[49]=12; cooColIndexHostPtr[49]=13; cooValHostPtr[49]=310.0731361794753;
    cooRowIndexHostPtr[50]=12; cooColIndexHostPtr[50]=112; cooValHostPtr[50]=4.0;
    cooRowIndexHostPtr[51]=13; cooColIndexHostPtr[51]=12; cooValHostPtr[51]=310.0731361794753;
    cooRowIndexHostPtr[52]=13; cooColIndexHostPtr[52]=13; cooValHostPtr[52]=-615.6962723589505;
    cooRowIndexHostPtr[53]=13; cooColIndexHostPtr[53]=14; cooValHostPtr[53]=310.0731361794753;
    cooRowIndexHostPtr[54]=13; cooColIndexHostPtr[54]=113; cooValHostPtr[54]=4.0;
    cooRowIndexHostPtr[55]=14; cooColIndexHostPtr[55]=13; cooValHostPtr[55]=310.0731361794753;
    cooRowIndexHostPtr[56]=14; cooColIndexHostPtr[56]=14; cooValHostPtr[56]=-615.6962723589505;
    cooRowIndexHostPtr[57]=14; cooColIndexHostPtr[57]=15; cooValHostPtr[57]=310.0731361794753;
    cooRowIndexHostPtr[58]=14; cooColIndexHostPtr[58]=114; cooValHostPtr[58]=4.0;
    cooRowIndexHostPtr[59]=15; cooColIndexHostPtr[59]=14; cooValHostPtr[59]=310.0731361794753;
    cooRowIndexHostPtr[60]=15; cooColIndexHostPtr[60]=15; cooValHostPtr[60]=-615.6962723589505;
    cooRowIndexHostPtr[61]=15; cooColIndexHostPtr[61]=16; cooValHostPtr[61]=310.0731361794753;
    cooRowIndexHostPtr[62]=15; cooColIndexHostPtr[62]=115; cooValHostPtr[62]=4.0;
    cooRowIndexHostPtr[63]=16; cooColIndexHostPtr[63]=15; cooValHostPtr[63]=310.0731361794753;
    cooRowIndexHostPtr[64]=16; cooColIndexHostPtr[64]=16; cooValHostPtr[64]=-615.6962723589505;
    cooRowIndexHostPtr[65]=16; cooColIndexHostPtr[65]=17; cooValHostPtr[65]=310.0731361794753;
    cooRowIndexHostPtr[66]=16; cooColIndexHostPtr[66]=116; cooValHostPtr[66]=4.0;
    cooRowIndexHostPtr[67]=17; cooColIndexHostPtr[67]=16; cooValHostPtr[67]=310.0731361794753;
    cooRowIndexHostPtr[68]=17; cooColIndexHostPtr[68]=17; cooValHostPtr[68]=-615.6962723589505;
    cooRowIndexHostPtr[69]=17; cooColIndexHostPtr[69]=18; cooValHostPtr[69]=310.0731361794753;
    cooRowIndexHostPtr[70]=17; cooColIndexHostPtr[70]=117; cooValHostPtr[70]=4.0;
    cooRowIndexHostPtr[71]=18; cooColIndexHostPtr[71]=17; cooValHostPtr[71]=310.0731361794753;
    cooRowIndexHostPtr[72]=18; cooColIndexHostPtr[72]=18; cooValHostPtr[72]=-615.6962723589505;
    cooRowIndexHostPtr[73]=18; cooColIndexHostPtr[73]=19; cooValHostPtr[73]=310.0731361794753;
    cooRowIndexHostPtr[74]=18; cooColIndexHostPtr[74]=118; cooValHostPtr[74]=4.0;
    cooRowIndexHostPtr[75]=19; cooColIndexHostPtr[75]=18; cooValHostPtr[75]=310.0731361794753;
    cooRowIndexHostPtr[76]=19; cooColIndexHostPtr[76]=19; cooValHostPtr[76]=-615.6962723589505;
    cooRowIndexHostPtr[77]=19; cooColIndexHostPtr[77]=20; cooValHostPtr[77]=310.0731361794753;
    cooRowIndexHostPtr[78]=19; cooColIndexHostPtr[78]=119; cooValHostPtr[78]=4.0;
    cooRowIndexHostPtr[79]=20; cooColIndexHostPtr[79]=19; cooValHostPtr[79]=310.0731361794753;
    cooRowIndexHostPtr[80]=20; cooColIndexHostPtr[80]=20; cooValHostPtr[80]=-615.6962723589505;
    cooRowIndexHostPtr[81]=20; cooColIndexHostPtr[81]=21; cooValHostPtr[81]=310.0731361794753;
    cooRowIndexHostPtr[82]=20; cooColIndexHostPtr[82]=120; cooValHostPtr[82]=4.0;
    cooRowIndexHostPtr[83]=21; cooColIndexHostPtr[83]=20; cooValHostPtr[83]=310.0731361794753;
    cooRowIndexHostPtr[84]=21; cooColIndexHostPtr[84]=21; cooValHostPtr[84]=-615.6962723589505;
    cooRowIndexHostPtr[85]=21; cooColIndexHostPtr[85]=22; cooValHostPtr[85]=310.0731361794753;
    cooRowIndexHostPtr[86]=21; cooColIndexHostPtr[86]=121; cooValHostPtr[86]=4.0;
    cooRowIndexHostPtr[87]=22; cooColIndexHostPtr[87]=21; cooValHostPtr[87]=310.0731361794753;
    cooRowIndexHostPtr[88]=22; cooColIndexHostPtr[88]=22; cooValHostPtr[88]=-615.6962723589505;
    cooRowIndexHostPtr[89]=22; cooColIndexHostPtr[89]=23; cooValHostPtr[89]=310.0731361794753;
    cooRowIndexHostPtr[90]=22; cooColIndexHostPtr[90]=122; cooValHostPtr[90]=4.0;
    cooRowIndexHostPtr[91]=23; cooColIndexHostPtr[91]=22; cooValHostPtr[91]=310.0731361794753;
    cooRowIndexHostPtr[92]=23; cooColIndexHostPtr[92]=23; cooValHostPtr[92]=-615.6962723589505;
    cooRowIndexHostPtr[93]=23; cooColIndexHostPtr[93]=24; cooValHostPtr[93]=310.0731361794753;
    cooRowIndexHostPtr[94]=23; cooColIndexHostPtr[94]=123; cooValHostPtr[94]=4.0;
    cooRowIndexHostPtr[95]=24; cooColIndexHostPtr[95]=23; cooValHostPtr[95]=310.0731361794753;
    cooRowIndexHostPtr[96]=24; cooColIndexHostPtr[96]=24; cooValHostPtr[96]=-615.6962723589505;
    cooRowIndexHostPtr[97]=24; cooColIndexHostPtr[97]=25; cooValHostPtr[97]=310.0731361794753;
    cooRowIndexHostPtr[98]=24; cooColIndexHostPtr[98]=124; cooValHostPtr[98]=4.0;
    cooRowIndexHostPtr[99]=25; cooColIndexHostPtr[99]=24; cooValHostPtr[99]=310.0731361794753;
    cooRowIndexHostPtr[100]=25; cooColIndexHostPtr[100]=25; cooValHostPtr[100]=-615.6962723589505;
    cooRowIndexHostPtr[101]=25; cooColIndexHostPtr[101]=26; cooValHostPtr[101]=310.0731361794753;
    cooRowIndexHostPtr[102]=25; cooColIndexHostPtr[102]=125; cooValHostPtr[102]=4.0;
    cooRowIndexHostPtr[103]=26; cooColIndexHostPtr[103]=25; cooValHostPtr[103]=310.0731361794753;
    cooRowIndexHostPtr[104]=26; cooColIndexHostPtr[104]=26; cooValHostPtr[104]=-615.6962723589505;
    cooRowIndexHostPtr[105]=26; cooColIndexHostPtr[105]=27; cooValHostPtr[105]=310.0731361794753;
    cooRowIndexHostPtr[106]=26; cooColIndexHostPtr[106]=126; cooValHostPtr[106]=4.0;
    cooRowIndexHostPtr[107]=27; cooColIndexHostPtr[107]=26; cooValHostPtr[107]=310.0731361794753;
    cooRowIndexHostPtr[108]=27; cooColIndexHostPtr[108]=27; cooValHostPtr[108]=-615.6962723589505;
    cooRowIndexHostPtr[109]=27; cooColIndexHostPtr[109]=28; cooValHostPtr[109]=310.0731361794753;
    cooRowIndexHostPtr[110]=27; cooColIndexHostPtr[110]=127; cooValHostPtr[110]=4.0;
    cooRowIndexHostPtr[111]=28; cooColIndexHostPtr[111]=27; cooValHostPtr[111]=310.0731361794753;
    cooRowIndexHostPtr[112]=28; cooColIndexHostPtr[112]=28; cooValHostPtr[112]=-615.6962723589505;
    cooRowIndexHostPtr[113]=28; cooColIndexHostPtr[113]=29; cooValHostPtr[113]=310.0731361794753;
    cooRowIndexHostPtr[114]=28; cooColIndexHostPtr[114]=128; cooValHostPtr[114]=4.0;
    cooRowIndexHostPtr[115]=29; cooColIndexHostPtr[115]=28; cooValHostPtr[115]=310.0731361794753;
    cooRowIndexHostPtr[116]=29; cooColIndexHostPtr[116]=29; cooValHostPtr[116]=-615.6962723589505;
    cooRowIndexHostPtr[117]=29; cooColIndexHostPtr[117]=30; cooValHostPtr[117]=310.0731361794753;
    cooRowIndexHostPtr[118]=29; cooColIndexHostPtr[118]=129; cooValHostPtr[118]=4.0;
    cooRowIndexHostPtr[119]=30; cooColIndexHostPtr[119]=29; cooValHostPtr[119]=310.0731361794753;
    cooRowIndexHostPtr[120]=30; cooColIndexHostPtr[120]=30; cooValHostPtr[120]=-615.6962723589505;
    cooRowIndexHostPtr[121]=30; cooColIndexHostPtr[121]=31; cooValHostPtr[121]=310.0731361794753;
    cooRowIndexHostPtr[122]=30; cooColIndexHostPtr[122]=130; cooValHostPtr[122]=4.0;
    cooRowIndexHostPtr[123]=31; cooColIndexHostPtr[123]=30; cooValHostPtr[123]=310.0731361794753;
    cooRowIndexHostPtr[124]=31; cooColIndexHostPtr[124]=31; cooValHostPtr[124]=-615.6962723589505;
    cooRowIndexHostPtr[125]=31; cooColIndexHostPtr[125]=32; cooValHostPtr[125]=310.0731361794753;
    cooRowIndexHostPtr[126]=31; cooColIndexHostPtr[126]=131; cooValHostPtr[126]=4.0;
    cooRowIndexHostPtr[127]=32; cooColIndexHostPtr[127]=31; cooValHostPtr[127]=310.0731361794753;
    cooRowIndexHostPtr[128]=32; cooColIndexHostPtr[128]=32; cooValHostPtr[128]=-615.6962723589505;
    cooRowIndexHostPtr[129]=32; cooColIndexHostPtr[129]=33; cooValHostPtr[129]=310.0731361794753;
    cooRowIndexHostPtr[130]=32; cooColIndexHostPtr[130]=132; cooValHostPtr[130]=4.0;
    cooRowIndexHostPtr[131]=33; cooColIndexHostPtr[131]=32; cooValHostPtr[131]=310.0731361794753;
    cooRowIndexHostPtr[132]=33; cooColIndexHostPtr[132]=33; cooValHostPtr[132]=-615.6962723589505;
    cooRowIndexHostPtr[133]=33; cooColIndexHostPtr[133]=34; cooValHostPtr[133]=310.0731361794753;
    cooRowIndexHostPtr[134]=33; cooColIndexHostPtr[134]=133; cooValHostPtr[134]=4.0;
    cooRowIndexHostPtr[135]=34; cooColIndexHostPtr[135]=33; cooValHostPtr[135]=310.0731361794753;
    cooRowIndexHostPtr[136]=34; cooColIndexHostPtr[136]=34; cooValHostPtr[136]=-615.6962723589505;
    cooRowIndexHostPtr[137]=34; cooColIndexHostPtr[137]=35; cooValHostPtr[137]=310.0731361794753;
    cooRowIndexHostPtr[138]=34; cooColIndexHostPtr[138]=134; cooValHostPtr[138]=4.0;
    cooRowIndexHostPtr[139]=35; cooColIndexHostPtr[139]=34; cooValHostPtr[139]=310.0731361794753;
    cooRowIndexHostPtr[140]=35; cooColIndexHostPtr[140]=35; cooValHostPtr[140]=-615.6962723589505;
    cooRowIndexHostPtr[141]=35; cooColIndexHostPtr[141]=36; cooValHostPtr[141]=310.0731361794753;
    cooRowIndexHostPtr[142]=35; cooColIndexHostPtr[142]=135; cooValHostPtr[142]=4.0;
    cooRowIndexHostPtr[143]=36; cooColIndexHostPtr[143]=35; cooValHostPtr[143]=310.0731361794753;
    cooRowIndexHostPtr[144]=36; cooColIndexHostPtr[144]=36; cooValHostPtr[144]=-615.6962723589505;
    cooRowIndexHostPtr[145]=36; cooColIndexHostPtr[145]=37; cooValHostPtr[145]=310.0731361794753;
    cooRowIndexHostPtr[146]=36; cooColIndexHostPtr[146]=136; cooValHostPtr[146]=4.0;
    cooRowIndexHostPtr[147]=37; cooColIndexHostPtr[147]=36; cooValHostPtr[147]=310.0731361794753;
    cooRowIndexHostPtr[148]=37; cooColIndexHostPtr[148]=37; cooValHostPtr[148]=-615.6962723589505;
    cooRowIndexHostPtr[149]=37; cooColIndexHostPtr[149]=38; cooValHostPtr[149]=310.0731361794753;
    cooRowIndexHostPtr[150]=37; cooColIndexHostPtr[150]=137; cooValHostPtr[150]=4.0;
    cooRowIndexHostPtr[151]=38; cooColIndexHostPtr[151]=37; cooValHostPtr[151]=310.0731361794753;
    cooRowIndexHostPtr[152]=38; cooColIndexHostPtr[152]=38; cooValHostPtr[152]=-615.6962723589505;
    cooRowIndexHostPtr[153]=38; cooColIndexHostPtr[153]=39; cooValHostPtr[153]=310.0731361794753;
    cooRowIndexHostPtr[154]=38; cooColIndexHostPtr[154]=138; cooValHostPtr[154]=4.0;
    cooRowIndexHostPtr[155]=39; cooColIndexHostPtr[155]=38; cooValHostPtr[155]=310.0731361794753;
    cooRowIndexHostPtr[156]=39; cooColIndexHostPtr[156]=39; cooValHostPtr[156]=-615.6962723589505;
    cooRowIndexHostPtr[157]=39; cooColIndexHostPtr[157]=40; cooValHostPtr[157]=310.0731361794753;
    cooRowIndexHostPtr[158]=39; cooColIndexHostPtr[158]=139; cooValHostPtr[158]=4.0;
    cooRowIndexHostPtr[159]=40; cooColIndexHostPtr[159]=39; cooValHostPtr[159]=310.0731361794753;
    cooRowIndexHostPtr[160]=40; cooColIndexHostPtr[160]=40; cooValHostPtr[160]=-615.6962723589505;
    cooRowIndexHostPtr[161]=40; cooColIndexHostPtr[161]=41; cooValHostPtr[161]=310.0731361794753;
    cooRowIndexHostPtr[162]=40; cooColIndexHostPtr[162]=140; cooValHostPtr[162]=4.0;
    cooRowIndexHostPtr[163]=41; cooColIndexHostPtr[163]=40; cooValHostPtr[163]=310.0731361794753;
    cooRowIndexHostPtr[164]=41; cooColIndexHostPtr[164]=41; cooValHostPtr[164]=-615.6962723589505;
    cooRowIndexHostPtr[165]=41; cooColIndexHostPtr[165]=42; cooValHostPtr[165]=310.0731361794753;
    cooRowIndexHostPtr[166]=41; cooColIndexHostPtr[166]=141; cooValHostPtr[166]=4.0;
    cooRowIndexHostPtr[167]=42; cooColIndexHostPtr[167]=41; cooValHostPtr[167]=310.0731361794753;
    cooRowIndexHostPtr[168]=42; cooColIndexHostPtr[168]=42; cooValHostPtr[168]=-615.6962723589505;
    cooRowIndexHostPtr[169]=42; cooColIndexHostPtr[169]=43; cooValHostPtr[169]=310.0731361794753;
    cooRowIndexHostPtr[170]=42; cooColIndexHostPtr[170]=142; cooValHostPtr[170]=4.0;
    cooRowIndexHostPtr[171]=43; cooColIndexHostPtr[171]=42; cooValHostPtr[171]=310.0731361794753;
    cooRowIndexHostPtr[172]=43; cooColIndexHostPtr[172]=43; cooValHostPtr[172]=-615.6962723589505;
    cooRowIndexHostPtr[173]=43; cooColIndexHostPtr[173]=44; cooValHostPtr[173]=310.0731361794753;
    cooRowIndexHostPtr[174]=43; cooColIndexHostPtr[174]=143; cooValHostPtr[174]=4.0;
    cooRowIndexHostPtr[175]=44; cooColIndexHostPtr[175]=43; cooValHostPtr[175]=310.0731361794753;
    cooRowIndexHostPtr[176]=44; cooColIndexHostPtr[176]=44; cooValHostPtr[176]=-615.6962723589505;
    cooRowIndexHostPtr[177]=44; cooColIndexHostPtr[177]=45; cooValHostPtr[177]=310.0731361794753;
    cooRowIndexHostPtr[178]=44; cooColIndexHostPtr[178]=144; cooValHostPtr[178]=4.0;
    cooRowIndexHostPtr[179]=45; cooColIndexHostPtr[179]=44; cooValHostPtr[179]=310.0731361794753;
    cooRowIndexHostPtr[180]=45; cooColIndexHostPtr[180]=45; cooValHostPtr[180]=-615.6962723589505;
    cooRowIndexHostPtr[181]=45; cooColIndexHostPtr[181]=46; cooValHostPtr[181]=310.0731361794753;
    cooRowIndexHostPtr[182]=45; cooColIndexHostPtr[182]=145; cooValHostPtr[182]=4.0;
    cooRowIndexHostPtr[183]=46; cooColIndexHostPtr[183]=45; cooValHostPtr[183]=310.0731361794753;
    cooRowIndexHostPtr[184]=46; cooColIndexHostPtr[184]=46; cooValHostPtr[184]=-615.6962723589505;
    cooRowIndexHostPtr[185]=46; cooColIndexHostPtr[185]=47; cooValHostPtr[185]=310.0731361794753;
    cooRowIndexHostPtr[186]=46; cooColIndexHostPtr[186]=146; cooValHostPtr[186]=4.0;
    cooRowIndexHostPtr[187]=47; cooColIndexHostPtr[187]=46; cooValHostPtr[187]=310.0731361794753;
    cooRowIndexHostPtr[188]=47; cooColIndexHostPtr[188]=47; cooValHostPtr[188]=-615.6962723589505;
    cooRowIndexHostPtr[189]=47; cooColIndexHostPtr[189]=48; cooValHostPtr[189]=310.0731361794753;
    cooRowIndexHostPtr[190]=47; cooColIndexHostPtr[190]=147; cooValHostPtr[190]=4.0;
    cooRowIndexHostPtr[191]=48; cooColIndexHostPtr[191]=47; cooValHostPtr[191]=310.0731361794753;
    cooRowIndexHostPtr[192]=48; cooColIndexHostPtr[192]=48; cooValHostPtr[192]=-615.6962723589505;
    cooRowIndexHostPtr[193]=48; cooColIndexHostPtr[193]=49; cooValHostPtr[193]=310.0731361794753;
    cooRowIndexHostPtr[194]=48; cooColIndexHostPtr[194]=148; cooValHostPtr[194]=4.0;
    cooRowIndexHostPtr[195]=49; cooColIndexHostPtr[195]=48; cooValHostPtr[195]=310.0731361794753;
    cooRowIndexHostPtr[196]=49; cooColIndexHostPtr[196]=49; cooValHostPtr[196]=-615.6962723589505;
    cooRowIndexHostPtr[197]=49; cooColIndexHostPtr[197]=50; cooValHostPtr[197]=310.0731361794753;
    cooRowIndexHostPtr[198]=49; cooColIndexHostPtr[198]=149; cooValHostPtr[198]=4.0;
    cooRowIndexHostPtr[199]=50; cooColIndexHostPtr[199]=49; cooValHostPtr[199]=310.0731361794753;
    cooRowIndexHostPtr[200]=50; cooColIndexHostPtr[200]=50; cooValHostPtr[200]=-615.6962723589505;
    cooRowIndexHostPtr[201]=50; cooColIndexHostPtr[201]=51; cooValHostPtr[201]=310.0731361794753;
    cooRowIndexHostPtr[202]=50; cooColIndexHostPtr[202]=150; cooValHostPtr[202]=4.0;
    cooRowIndexHostPtr[203]=51; cooColIndexHostPtr[203]=50; cooValHostPtr[203]=310.0731361794753;
    cooRowIndexHostPtr[204]=51; cooColIndexHostPtr[204]=51; cooValHostPtr[204]=-615.6962723589505;
    cooRowIndexHostPtr[205]=51; cooColIndexHostPtr[205]=52; cooValHostPtr[205]=310.0731361794753;
    cooRowIndexHostPtr[206]=51; cooColIndexHostPtr[206]=151; cooValHostPtr[206]=4.0;
    cooRowIndexHostPtr[207]=52; cooColIndexHostPtr[207]=51; cooValHostPtr[207]=310.0731361794753;
    cooRowIndexHostPtr[208]=52; cooColIndexHostPtr[208]=52; cooValHostPtr[208]=-615.6962723589505;
    cooRowIndexHostPtr[209]=52; cooColIndexHostPtr[209]=53; cooValHostPtr[209]=310.0731361794753;
    cooRowIndexHostPtr[210]=52; cooColIndexHostPtr[210]=152; cooValHostPtr[210]=4.0;
    cooRowIndexHostPtr[211]=53; cooColIndexHostPtr[211]=52; cooValHostPtr[211]=310.0731361794753;
    cooRowIndexHostPtr[212]=53; cooColIndexHostPtr[212]=53; cooValHostPtr[212]=-615.6962723589505;
    cooRowIndexHostPtr[213]=53; cooColIndexHostPtr[213]=54; cooValHostPtr[213]=310.0731361794753;
    cooRowIndexHostPtr[214]=53; cooColIndexHostPtr[214]=153; cooValHostPtr[214]=4.0;
    cooRowIndexHostPtr[215]=54; cooColIndexHostPtr[215]=53; cooValHostPtr[215]=310.0731361794753;
    cooRowIndexHostPtr[216]=54; cooColIndexHostPtr[216]=54; cooValHostPtr[216]=-615.6962723589505;
    cooRowIndexHostPtr[217]=54; cooColIndexHostPtr[217]=55; cooValHostPtr[217]=310.0731361794753;
    cooRowIndexHostPtr[218]=54; cooColIndexHostPtr[218]=154; cooValHostPtr[218]=4.0;
    cooRowIndexHostPtr[219]=55; cooColIndexHostPtr[219]=54; cooValHostPtr[219]=310.0731361794753;
    cooRowIndexHostPtr[220]=55; cooColIndexHostPtr[220]=55; cooValHostPtr[220]=-615.6962723589505;
    cooRowIndexHostPtr[221]=55; cooColIndexHostPtr[221]=56; cooValHostPtr[221]=310.0731361794753;
    cooRowIndexHostPtr[222]=55; cooColIndexHostPtr[222]=155; cooValHostPtr[222]=4.0;
    cooRowIndexHostPtr[223]=56; cooColIndexHostPtr[223]=55; cooValHostPtr[223]=310.0731361794753;
    cooRowIndexHostPtr[224]=56; cooColIndexHostPtr[224]=56; cooValHostPtr[224]=-615.6962723589505;
    cooRowIndexHostPtr[225]=56; cooColIndexHostPtr[225]=57; cooValHostPtr[225]=310.0731361794753;
    cooRowIndexHostPtr[226]=56; cooColIndexHostPtr[226]=156; cooValHostPtr[226]=4.0;
    cooRowIndexHostPtr[227]=57; cooColIndexHostPtr[227]=56; cooValHostPtr[227]=310.0731361794753;
    cooRowIndexHostPtr[228]=57; cooColIndexHostPtr[228]=57; cooValHostPtr[228]=-615.6962723589505;
    cooRowIndexHostPtr[229]=57; cooColIndexHostPtr[229]=58; cooValHostPtr[229]=310.0731361794753;
    cooRowIndexHostPtr[230]=57; cooColIndexHostPtr[230]=157; cooValHostPtr[230]=4.0;
    cooRowIndexHostPtr[231]=58; cooColIndexHostPtr[231]=57; cooValHostPtr[231]=310.0731361794753;
    cooRowIndexHostPtr[232]=58; cooColIndexHostPtr[232]=58; cooValHostPtr[232]=-615.6962723589505;
    cooRowIndexHostPtr[233]=58; cooColIndexHostPtr[233]=59; cooValHostPtr[233]=310.0731361794753;
    cooRowIndexHostPtr[234]=58; cooColIndexHostPtr[234]=158; cooValHostPtr[234]=4.0;
    cooRowIndexHostPtr[235]=59; cooColIndexHostPtr[235]=58; cooValHostPtr[235]=310.0731361794753;
    cooRowIndexHostPtr[236]=59; cooColIndexHostPtr[236]=59; cooValHostPtr[236]=-615.6962723589505;
    cooRowIndexHostPtr[237]=59; cooColIndexHostPtr[237]=60; cooValHostPtr[237]=310.0731361794753;
    cooRowIndexHostPtr[238]=59; cooColIndexHostPtr[238]=159; cooValHostPtr[238]=4.0;
    cooRowIndexHostPtr[239]=60; cooColIndexHostPtr[239]=59; cooValHostPtr[239]=310.0731361794753;
    cooRowIndexHostPtr[240]=60; cooColIndexHostPtr[240]=60; cooValHostPtr[240]=-615.6962723589505;
    cooRowIndexHostPtr[241]=60; cooColIndexHostPtr[241]=61; cooValHostPtr[241]=310.0731361794753;
    cooRowIndexHostPtr[242]=60; cooColIndexHostPtr[242]=160; cooValHostPtr[242]=4.0;
    cooRowIndexHostPtr[243]=61; cooColIndexHostPtr[243]=60; cooValHostPtr[243]=310.0731361794753;
    cooRowIndexHostPtr[244]=61; cooColIndexHostPtr[244]=61; cooValHostPtr[244]=-615.6962723589505;
    cooRowIndexHostPtr[245]=61; cooColIndexHostPtr[245]=62; cooValHostPtr[245]=310.0731361794753;
    cooRowIndexHostPtr[246]=61; cooColIndexHostPtr[246]=161; cooValHostPtr[246]=4.0;
    cooRowIndexHostPtr[247]=62; cooColIndexHostPtr[247]=61; cooValHostPtr[247]=310.0731361794753;
    cooRowIndexHostPtr[248]=62; cooColIndexHostPtr[248]=62; cooValHostPtr[248]=-615.6962723589505;
    cooRowIndexHostPtr[249]=62; cooColIndexHostPtr[249]=63; cooValHostPtr[249]=310.0731361794753;
    cooRowIndexHostPtr[250]=62; cooColIndexHostPtr[250]=162; cooValHostPtr[250]=4.0;
    cooRowIndexHostPtr[251]=63; cooColIndexHostPtr[251]=62; cooValHostPtr[251]=310.0731361794753;
    cooRowIndexHostPtr[252]=63; cooColIndexHostPtr[252]=63; cooValHostPtr[252]=-615.6962723589505;
    cooRowIndexHostPtr[253]=63; cooColIndexHostPtr[253]=64; cooValHostPtr[253]=310.0731361794753;
    cooRowIndexHostPtr[254]=63; cooColIndexHostPtr[254]=163; cooValHostPtr[254]=4.0;
    cooRowIndexHostPtr[255]=64; cooColIndexHostPtr[255]=63; cooValHostPtr[255]=310.0731361794753;
    cooRowIndexHostPtr[256]=64; cooColIndexHostPtr[256]=64; cooValHostPtr[256]=-615.6962723589505;
    cooRowIndexHostPtr[257]=64; cooColIndexHostPtr[257]=65; cooValHostPtr[257]=310.0731361794753;
    cooRowIndexHostPtr[258]=64; cooColIndexHostPtr[258]=164; cooValHostPtr[258]=4.0;
    cooRowIndexHostPtr[259]=65; cooColIndexHostPtr[259]=64; cooValHostPtr[259]=310.0731361794753;
    cooRowIndexHostPtr[260]=65; cooColIndexHostPtr[260]=65; cooValHostPtr[260]=-615.6962723589505;
    cooRowIndexHostPtr[261]=65; cooColIndexHostPtr[261]=66; cooValHostPtr[261]=310.0731361794753;
    cooRowIndexHostPtr[262]=65; cooColIndexHostPtr[262]=165; cooValHostPtr[262]=4.0;
    cooRowIndexHostPtr[263]=66; cooColIndexHostPtr[263]=65; cooValHostPtr[263]=310.0731361794753;
    cooRowIndexHostPtr[264]=66; cooColIndexHostPtr[264]=66; cooValHostPtr[264]=-615.6962723589505;
    cooRowIndexHostPtr[265]=66; cooColIndexHostPtr[265]=67; cooValHostPtr[265]=310.0731361794753;
    cooRowIndexHostPtr[266]=66; cooColIndexHostPtr[266]=166; cooValHostPtr[266]=4.0;
    cooRowIndexHostPtr[267]=67; cooColIndexHostPtr[267]=66; cooValHostPtr[267]=310.0731361794753;
    cooRowIndexHostPtr[268]=67; cooColIndexHostPtr[268]=67; cooValHostPtr[268]=-615.6962723589505;
    cooRowIndexHostPtr[269]=67; cooColIndexHostPtr[269]=68; cooValHostPtr[269]=310.0731361794753;
    cooRowIndexHostPtr[270]=67; cooColIndexHostPtr[270]=167; cooValHostPtr[270]=4.0;
    cooRowIndexHostPtr[271]=68; cooColIndexHostPtr[271]=67; cooValHostPtr[271]=310.0731361794753;
    cooRowIndexHostPtr[272]=68; cooColIndexHostPtr[272]=68; cooValHostPtr[272]=-615.6962723589505;
    cooRowIndexHostPtr[273]=68; cooColIndexHostPtr[273]=69; cooValHostPtr[273]=310.0731361794753;
    cooRowIndexHostPtr[274]=68; cooColIndexHostPtr[274]=168; cooValHostPtr[274]=4.0;
    cooRowIndexHostPtr[275]=69; cooColIndexHostPtr[275]=68; cooValHostPtr[275]=310.0731361794753;
    cooRowIndexHostPtr[276]=69; cooColIndexHostPtr[276]=69; cooValHostPtr[276]=-615.6962723589505;
    cooRowIndexHostPtr[277]=69; cooColIndexHostPtr[277]=70; cooValHostPtr[277]=310.0731361794753;
    cooRowIndexHostPtr[278]=69; cooColIndexHostPtr[278]=169; cooValHostPtr[278]=4.0;
    cooRowIndexHostPtr[279]=70; cooColIndexHostPtr[279]=69; cooValHostPtr[279]=310.0731361794753;
    cooRowIndexHostPtr[280]=70; cooColIndexHostPtr[280]=70; cooValHostPtr[280]=-615.6962723589505;
    cooRowIndexHostPtr[281]=70; cooColIndexHostPtr[281]=71; cooValHostPtr[281]=310.0731361794753;
    cooRowIndexHostPtr[282]=70; cooColIndexHostPtr[282]=170; cooValHostPtr[282]=4.0;
    cooRowIndexHostPtr[283]=71; cooColIndexHostPtr[283]=70; cooValHostPtr[283]=310.0731361794753;
    cooRowIndexHostPtr[284]=71; cooColIndexHostPtr[284]=71; cooValHostPtr[284]=-615.6962723589505;
    cooRowIndexHostPtr[285]=71; cooColIndexHostPtr[285]=72; cooValHostPtr[285]=310.0731361794753;
    cooRowIndexHostPtr[286]=71; cooColIndexHostPtr[286]=171; cooValHostPtr[286]=4.0;
    cooRowIndexHostPtr[287]=72; cooColIndexHostPtr[287]=71; cooValHostPtr[287]=310.0731361794753;
    cooRowIndexHostPtr[288]=72; cooColIndexHostPtr[288]=72; cooValHostPtr[288]=-615.6962723589505;
    cooRowIndexHostPtr[289]=72; cooColIndexHostPtr[289]=73; cooValHostPtr[289]=310.0731361794753;
    cooRowIndexHostPtr[290]=72; cooColIndexHostPtr[290]=172; cooValHostPtr[290]=4.0;
    cooRowIndexHostPtr[291]=73; cooColIndexHostPtr[291]=72; cooValHostPtr[291]=310.0731361794753;
    cooRowIndexHostPtr[292]=73; cooColIndexHostPtr[292]=73; cooValHostPtr[292]=-615.6962723589505;
    cooRowIndexHostPtr[293]=73; cooColIndexHostPtr[293]=74; cooValHostPtr[293]=310.0731361794753;
    cooRowIndexHostPtr[294]=73; cooColIndexHostPtr[294]=173; cooValHostPtr[294]=4.0;
    cooRowIndexHostPtr[295]=74; cooColIndexHostPtr[295]=73; cooValHostPtr[295]=310.0731361794753;
    cooRowIndexHostPtr[296]=74; cooColIndexHostPtr[296]=74; cooValHostPtr[296]=-615.6962723589505;
    cooRowIndexHostPtr[297]=74; cooColIndexHostPtr[297]=75; cooValHostPtr[297]=310.0731361794753;
    cooRowIndexHostPtr[298]=74; cooColIndexHostPtr[298]=174; cooValHostPtr[298]=4.0;
    cooRowIndexHostPtr[299]=75; cooColIndexHostPtr[299]=74; cooValHostPtr[299]=310.0731361794753;
    cooRowIndexHostPtr[300]=75; cooColIndexHostPtr[300]=75; cooValHostPtr[300]=-615.6962723589505;
    cooRowIndexHostPtr[301]=75; cooColIndexHostPtr[301]=76; cooValHostPtr[301]=310.0731361794753;
    cooRowIndexHostPtr[302]=75; cooColIndexHostPtr[302]=175; cooValHostPtr[302]=4.0;
    cooRowIndexHostPtr[303]=76; cooColIndexHostPtr[303]=75; cooValHostPtr[303]=310.0731361794753;
    cooRowIndexHostPtr[304]=76; cooColIndexHostPtr[304]=76; cooValHostPtr[304]=-615.6962723589505;
    cooRowIndexHostPtr[305]=76; cooColIndexHostPtr[305]=77; cooValHostPtr[305]=310.0731361794753;
    cooRowIndexHostPtr[306]=76; cooColIndexHostPtr[306]=176; cooValHostPtr[306]=4.0;
    cooRowIndexHostPtr[307]=77; cooColIndexHostPtr[307]=76; cooValHostPtr[307]=310.0731361794753;
    cooRowIndexHostPtr[308]=77; cooColIndexHostPtr[308]=77; cooValHostPtr[308]=-615.6962723589505;
    cooRowIndexHostPtr[309]=77; cooColIndexHostPtr[309]=78; cooValHostPtr[309]=310.0731361794753;
    cooRowIndexHostPtr[310]=77; cooColIndexHostPtr[310]=177; cooValHostPtr[310]=4.0;
    cooRowIndexHostPtr[311]=78; cooColIndexHostPtr[311]=77; cooValHostPtr[311]=310.0731361794753;
    cooRowIndexHostPtr[312]=78; cooColIndexHostPtr[312]=78; cooValHostPtr[312]=-615.6962723589505;
    cooRowIndexHostPtr[313]=78; cooColIndexHostPtr[313]=79; cooValHostPtr[313]=310.0731361794753;
    cooRowIndexHostPtr[314]=78; cooColIndexHostPtr[314]=178; cooValHostPtr[314]=4.0;
    cooRowIndexHostPtr[315]=79; cooColIndexHostPtr[315]=78; cooValHostPtr[315]=310.0731361794753;
    cooRowIndexHostPtr[316]=79; cooColIndexHostPtr[316]=79; cooValHostPtr[316]=-615.6962723589505;
    cooRowIndexHostPtr[317]=79; cooColIndexHostPtr[317]=80; cooValHostPtr[317]=310.0731361794753;
    cooRowIndexHostPtr[318]=79; cooColIndexHostPtr[318]=179; cooValHostPtr[318]=4.0;
    cooRowIndexHostPtr[319]=80; cooColIndexHostPtr[319]=79; cooValHostPtr[319]=310.0731361794753;
    cooRowIndexHostPtr[320]=80; cooColIndexHostPtr[320]=80; cooValHostPtr[320]=-615.6962723589505;
    cooRowIndexHostPtr[321]=80; cooColIndexHostPtr[321]=81; cooValHostPtr[321]=310.0731361794753;
    cooRowIndexHostPtr[322]=80; cooColIndexHostPtr[322]=180; cooValHostPtr[322]=4.0;
    cooRowIndexHostPtr[323]=81; cooColIndexHostPtr[323]=80; cooValHostPtr[323]=310.0731361794753;
    cooRowIndexHostPtr[324]=81; cooColIndexHostPtr[324]=81; cooValHostPtr[324]=-615.6962723589505;
    cooRowIndexHostPtr[325]=81; cooColIndexHostPtr[325]=82; cooValHostPtr[325]=310.0731361794753;
    cooRowIndexHostPtr[326]=81; cooColIndexHostPtr[326]=181; cooValHostPtr[326]=4.0;
    cooRowIndexHostPtr[327]=82; cooColIndexHostPtr[327]=81; cooValHostPtr[327]=310.0731361794753;
    cooRowIndexHostPtr[328]=82; cooColIndexHostPtr[328]=82; cooValHostPtr[328]=-615.6962723589505;
    cooRowIndexHostPtr[329]=82; cooColIndexHostPtr[329]=83; cooValHostPtr[329]=310.0731361794753;
    cooRowIndexHostPtr[330]=82; cooColIndexHostPtr[330]=182; cooValHostPtr[330]=4.0;
    cooRowIndexHostPtr[331]=83; cooColIndexHostPtr[331]=82; cooValHostPtr[331]=310.0731361794753;
    cooRowIndexHostPtr[332]=83; cooColIndexHostPtr[332]=83; cooValHostPtr[332]=-615.6962723589505;
    cooRowIndexHostPtr[333]=83; cooColIndexHostPtr[333]=84; cooValHostPtr[333]=310.0731361794753;
    cooRowIndexHostPtr[334]=83; cooColIndexHostPtr[334]=183; cooValHostPtr[334]=4.0;
    cooRowIndexHostPtr[335]=84; cooColIndexHostPtr[335]=83; cooValHostPtr[335]=310.0731361794753;
    cooRowIndexHostPtr[336]=84; cooColIndexHostPtr[336]=84; cooValHostPtr[336]=-615.6962723589505;
    cooRowIndexHostPtr[337]=84; cooColIndexHostPtr[337]=85; cooValHostPtr[337]=310.0731361794753;
    cooRowIndexHostPtr[338]=84; cooColIndexHostPtr[338]=184; cooValHostPtr[338]=4.0;
    cooRowIndexHostPtr[339]=85; cooColIndexHostPtr[339]=84; cooValHostPtr[339]=310.0731361794753;
    cooRowIndexHostPtr[340]=85; cooColIndexHostPtr[340]=85; cooValHostPtr[340]=-615.6962723589505;
    cooRowIndexHostPtr[341]=85; cooColIndexHostPtr[341]=86; cooValHostPtr[341]=310.0731361794753;
    cooRowIndexHostPtr[342]=85; cooColIndexHostPtr[342]=185; cooValHostPtr[342]=4.0;
    cooRowIndexHostPtr[343]=86; cooColIndexHostPtr[343]=85; cooValHostPtr[343]=310.0731361794753;
    cooRowIndexHostPtr[344]=86; cooColIndexHostPtr[344]=86; cooValHostPtr[344]=-615.6962723589505;
    cooRowIndexHostPtr[345]=86; cooColIndexHostPtr[345]=87; cooValHostPtr[345]=310.0731361794753;
    cooRowIndexHostPtr[346]=86; cooColIndexHostPtr[346]=186; cooValHostPtr[346]=4.0;
    cooRowIndexHostPtr[347]=87; cooColIndexHostPtr[347]=86; cooValHostPtr[347]=310.0731361794753;
    cooRowIndexHostPtr[348]=87; cooColIndexHostPtr[348]=87; cooValHostPtr[348]=-615.6962723589505;
    cooRowIndexHostPtr[349]=87; cooColIndexHostPtr[349]=88; cooValHostPtr[349]=310.0731361794753;
    cooRowIndexHostPtr[350]=87; cooColIndexHostPtr[350]=187; cooValHostPtr[350]=4.0;
    cooRowIndexHostPtr[351]=88; cooColIndexHostPtr[351]=87; cooValHostPtr[351]=310.0731361794753;
    cooRowIndexHostPtr[352]=88; cooColIndexHostPtr[352]=88; cooValHostPtr[352]=-615.6962723589505;
    cooRowIndexHostPtr[353]=88; cooColIndexHostPtr[353]=89; cooValHostPtr[353]=310.0731361794753;
    cooRowIndexHostPtr[354]=88; cooColIndexHostPtr[354]=188; cooValHostPtr[354]=4.0;
    cooRowIndexHostPtr[355]=89; cooColIndexHostPtr[355]=88; cooValHostPtr[355]=310.0731361794753;
    cooRowIndexHostPtr[356]=89; cooColIndexHostPtr[356]=89; cooValHostPtr[356]=-615.6962723589505;
    cooRowIndexHostPtr[357]=89; cooColIndexHostPtr[357]=90; cooValHostPtr[357]=310.0731361794753;
    cooRowIndexHostPtr[358]=89; cooColIndexHostPtr[358]=189; cooValHostPtr[358]=4.0;
    cooRowIndexHostPtr[359]=90; cooColIndexHostPtr[359]=89; cooValHostPtr[359]=310.0731361794753;
    cooRowIndexHostPtr[360]=90; cooColIndexHostPtr[360]=90; cooValHostPtr[360]=-615.6962723589505;
    cooRowIndexHostPtr[361]=90; cooColIndexHostPtr[361]=91; cooValHostPtr[361]=310.0731361794753;
    cooRowIndexHostPtr[362]=90; cooColIndexHostPtr[362]=190; cooValHostPtr[362]=4.0;
    cooRowIndexHostPtr[363]=91; cooColIndexHostPtr[363]=90; cooValHostPtr[363]=310.0731361794753;
    cooRowIndexHostPtr[364]=91; cooColIndexHostPtr[364]=91; cooValHostPtr[364]=-615.6962723589505;
    cooRowIndexHostPtr[365]=91; cooColIndexHostPtr[365]=92; cooValHostPtr[365]=310.0731361794753;
    cooRowIndexHostPtr[366]=91; cooColIndexHostPtr[366]=191; cooValHostPtr[366]=4.0;
    cooRowIndexHostPtr[367]=92; cooColIndexHostPtr[367]=91; cooValHostPtr[367]=310.0731361794753;
    cooRowIndexHostPtr[368]=92; cooColIndexHostPtr[368]=92; cooValHostPtr[368]=-615.6962723589505;
    cooRowIndexHostPtr[369]=92; cooColIndexHostPtr[369]=93; cooValHostPtr[369]=310.0731361794753;
    cooRowIndexHostPtr[370]=92; cooColIndexHostPtr[370]=192; cooValHostPtr[370]=4.0;
    cooRowIndexHostPtr[371]=93; cooColIndexHostPtr[371]=92; cooValHostPtr[371]=310.0731361794753;
    cooRowIndexHostPtr[372]=93; cooColIndexHostPtr[372]=93; cooValHostPtr[372]=-615.6962723589505;
    cooRowIndexHostPtr[373]=93; cooColIndexHostPtr[373]=94; cooValHostPtr[373]=310.0731361794753;
    cooRowIndexHostPtr[374]=93; cooColIndexHostPtr[374]=193; cooValHostPtr[374]=4.0;
    cooRowIndexHostPtr[375]=94; cooColIndexHostPtr[375]=93; cooValHostPtr[375]=310.0731361794753;
    cooRowIndexHostPtr[376]=94; cooColIndexHostPtr[376]=94; cooValHostPtr[376]=-615.6962723589505;
    cooRowIndexHostPtr[377]=94; cooColIndexHostPtr[377]=95; cooValHostPtr[377]=310.0731361794753;
    cooRowIndexHostPtr[378]=94; cooColIndexHostPtr[378]=194; cooValHostPtr[378]=4.0;
    cooRowIndexHostPtr[379]=95; cooColIndexHostPtr[379]=94; cooValHostPtr[379]=310.0731361794753;
    cooRowIndexHostPtr[380]=95; cooColIndexHostPtr[380]=95; cooValHostPtr[380]=-615.6962723589505;
    cooRowIndexHostPtr[381]=95; cooColIndexHostPtr[381]=96; cooValHostPtr[381]=310.0731361794753;
    cooRowIndexHostPtr[382]=95; cooColIndexHostPtr[382]=195; cooValHostPtr[382]=4.0;
    cooRowIndexHostPtr[383]=96; cooColIndexHostPtr[383]=95; cooValHostPtr[383]=310.0731361794753;
    cooRowIndexHostPtr[384]=96; cooColIndexHostPtr[384]=96; cooValHostPtr[384]=-615.6962723589505;
    cooRowIndexHostPtr[385]=96; cooColIndexHostPtr[385]=97; cooValHostPtr[385]=310.0731361794753;
    cooRowIndexHostPtr[386]=96; cooColIndexHostPtr[386]=196; cooValHostPtr[386]=4.0;
    cooRowIndexHostPtr[387]=97; cooColIndexHostPtr[387]=96; cooValHostPtr[387]=310.0731361794753;
    cooRowIndexHostPtr[388]=97; cooColIndexHostPtr[388]=97; cooValHostPtr[388]=-615.6962723589505;
    cooRowIndexHostPtr[389]=97; cooColIndexHostPtr[389]=98; cooValHostPtr[389]=310.0731361794753;
    cooRowIndexHostPtr[390]=97; cooColIndexHostPtr[390]=197; cooValHostPtr[390]=4.0;
    cooRowIndexHostPtr[391]=98; cooColIndexHostPtr[391]=97; cooValHostPtr[391]=310.0731361794753;
    cooRowIndexHostPtr[392]=98; cooColIndexHostPtr[392]=98; cooValHostPtr[392]=-615.6962723589505;
    cooRowIndexHostPtr[393]=98; cooColIndexHostPtr[393]=99; cooValHostPtr[393]=310.0731361794753;
    cooRowIndexHostPtr[394]=98; cooColIndexHostPtr[394]=198; cooValHostPtr[394]=4.0;
    cooRowIndexHostPtr[395]=99; cooColIndexHostPtr[395]=98; cooValHostPtr[395]=310.0731361794753;
    cooRowIndexHostPtr[396]=99; cooColIndexHostPtr[396]=99; cooValHostPtr[396]=-615.6962723589505;
    cooRowIndexHostPtr[397]=99; cooColIndexHostPtr[397]=199; cooValHostPtr[397]=4.0;
    cooRowIndexHostPtr[398]=100; cooColIndexHostPtr[398]=0; cooValHostPtr[398]=-5.45;
    cooRowIndexHostPtr[399]=100; cooColIndexHostPtr[399]=100; cooValHostPtr[399]=-314.0731361794753;
    cooRowIndexHostPtr[400]=100; cooColIndexHostPtr[400]=101; cooValHostPtr[400]=155.0365680897376;
    cooRowIndexHostPtr[401]=101; cooColIndexHostPtr[401]=1; cooValHostPtr[401]=-5.45;
    cooRowIndexHostPtr[402]=101; cooColIndexHostPtr[402]=100; cooValHostPtr[402]=155.0365680897376;
    cooRowIndexHostPtr[403]=101; cooColIndexHostPtr[403]=101; cooValHostPtr[403]=-314.0731361794753;
    cooRowIndexHostPtr[404]=101; cooColIndexHostPtr[404]=102; cooValHostPtr[404]=155.0365680897376;
    cooRowIndexHostPtr[405]=102; cooColIndexHostPtr[405]=2; cooValHostPtr[405]=-5.45;
    cooRowIndexHostPtr[406]=102; cooColIndexHostPtr[406]=101; cooValHostPtr[406]=155.0365680897376;
    cooRowIndexHostPtr[407]=102; cooColIndexHostPtr[407]=102; cooValHostPtr[407]=-314.0731361794753;
    cooRowIndexHostPtr[408]=102; cooColIndexHostPtr[408]=103; cooValHostPtr[408]=155.0365680897376;
    cooRowIndexHostPtr[409]=103; cooColIndexHostPtr[409]=3; cooValHostPtr[409]=-5.45;
    cooRowIndexHostPtr[410]=103; cooColIndexHostPtr[410]=102; cooValHostPtr[410]=155.0365680897376;
    cooRowIndexHostPtr[411]=103; cooColIndexHostPtr[411]=103; cooValHostPtr[411]=-314.0731361794753;
    cooRowIndexHostPtr[412]=103; cooColIndexHostPtr[412]=104; cooValHostPtr[412]=155.0365680897376;
    cooRowIndexHostPtr[413]=104; cooColIndexHostPtr[413]=4; cooValHostPtr[413]=-5.45;
    cooRowIndexHostPtr[414]=104; cooColIndexHostPtr[414]=103; cooValHostPtr[414]=155.0365680897376;
    cooRowIndexHostPtr[415]=104; cooColIndexHostPtr[415]=104; cooValHostPtr[415]=-314.0731361794753;
    cooRowIndexHostPtr[416]=104; cooColIndexHostPtr[416]=105; cooValHostPtr[416]=155.0365680897376;
    cooRowIndexHostPtr[417]=105; cooColIndexHostPtr[417]=5; cooValHostPtr[417]=-5.45;
    cooRowIndexHostPtr[418]=105; cooColIndexHostPtr[418]=104; cooValHostPtr[418]=155.0365680897376;
    cooRowIndexHostPtr[419]=105; cooColIndexHostPtr[419]=105; cooValHostPtr[419]=-314.0731361794753;
    cooRowIndexHostPtr[420]=105; cooColIndexHostPtr[420]=106; cooValHostPtr[420]=155.0365680897376;
    cooRowIndexHostPtr[421]=106; cooColIndexHostPtr[421]=6; cooValHostPtr[421]=-5.45;
    cooRowIndexHostPtr[422]=106; cooColIndexHostPtr[422]=105; cooValHostPtr[422]=155.0365680897376;
    cooRowIndexHostPtr[423]=106; cooColIndexHostPtr[423]=106; cooValHostPtr[423]=-314.0731361794753;
    cooRowIndexHostPtr[424]=106; cooColIndexHostPtr[424]=107; cooValHostPtr[424]=155.0365680897376;
    cooRowIndexHostPtr[425]=107; cooColIndexHostPtr[425]=7; cooValHostPtr[425]=-5.45;
    cooRowIndexHostPtr[426]=107; cooColIndexHostPtr[426]=106; cooValHostPtr[426]=155.0365680897376;
    cooRowIndexHostPtr[427]=107; cooColIndexHostPtr[427]=107; cooValHostPtr[427]=-314.0731361794753;
    cooRowIndexHostPtr[428]=107; cooColIndexHostPtr[428]=108; cooValHostPtr[428]=155.0365680897376;
    cooRowIndexHostPtr[429]=108; cooColIndexHostPtr[429]=8; cooValHostPtr[429]=-5.45;
    cooRowIndexHostPtr[430]=108; cooColIndexHostPtr[430]=107; cooValHostPtr[430]=155.0365680897376;
    cooRowIndexHostPtr[431]=108; cooColIndexHostPtr[431]=108; cooValHostPtr[431]=-314.0731361794753;
    cooRowIndexHostPtr[432]=108; cooColIndexHostPtr[432]=109; cooValHostPtr[432]=155.0365680897376;
    cooRowIndexHostPtr[433]=109; cooColIndexHostPtr[433]=9; cooValHostPtr[433]=-5.45;
    cooRowIndexHostPtr[434]=109; cooColIndexHostPtr[434]=108; cooValHostPtr[434]=155.0365680897376;
    cooRowIndexHostPtr[435]=109; cooColIndexHostPtr[435]=109; cooValHostPtr[435]=-314.0731361794753;
    cooRowIndexHostPtr[436]=109; cooColIndexHostPtr[436]=110; cooValHostPtr[436]=155.0365680897376;
    cooRowIndexHostPtr[437]=110; cooColIndexHostPtr[437]=10; cooValHostPtr[437]=-5.45;
    cooRowIndexHostPtr[438]=110; cooColIndexHostPtr[438]=109; cooValHostPtr[438]=155.0365680897376;
    cooRowIndexHostPtr[439]=110; cooColIndexHostPtr[439]=110; cooValHostPtr[439]=-314.0731361794753;
    cooRowIndexHostPtr[440]=110; cooColIndexHostPtr[440]=111; cooValHostPtr[440]=155.0365680897376;
    cooRowIndexHostPtr[441]=111; cooColIndexHostPtr[441]=11; cooValHostPtr[441]=-5.45;
    cooRowIndexHostPtr[442]=111; cooColIndexHostPtr[442]=110; cooValHostPtr[442]=155.0365680897376;
    cooRowIndexHostPtr[443]=111; cooColIndexHostPtr[443]=111; cooValHostPtr[443]=-314.0731361794753;
    cooRowIndexHostPtr[444]=111; cooColIndexHostPtr[444]=112; cooValHostPtr[444]=155.0365680897376;
    cooRowIndexHostPtr[445]=112; cooColIndexHostPtr[445]=12; cooValHostPtr[445]=-5.45;
    cooRowIndexHostPtr[446]=112; cooColIndexHostPtr[446]=111; cooValHostPtr[446]=155.0365680897376;
    cooRowIndexHostPtr[447]=112; cooColIndexHostPtr[447]=112; cooValHostPtr[447]=-314.0731361794753;
    cooRowIndexHostPtr[448]=112; cooColIndexHostPtr[448]=113; cooValHostPtr[448]=155.0365680897376;
    cooRowIndexHostPtr[449]=113; cooColIndexHostPtr[449]=13; cooValHostPtr[449]=-5.45;
    cooRowIndexHostPtr[450]=113; cooColIndexHostPtr[450]=112; cooValHostPtr[450]=155.0365680897376;
    cooRowIndexHostPtr[451]=113; cooColIndexHostPtr[451]=113; cooValHostPtr[451]=-314.0731361794753;
    cooRowIndexHostPtr[452]=113; cooColIndexHostPtr[452]=114; cooValHostPtr[452]=155.0365680897376;
    cooRowIndexHostPtr[453]=114; cooColIndexHostPtr[453]=14; cooValHostPtr[453]=-5.45;
    cooRowIndexHostPtr[454]=114; cooColIndexHostPtr[454]=113; cooValHostPtr[454]=155.0365680897376;
    cooRowIndexHostPtr[455]=114; cooColIndexHostPtr[455]=114; cooValHostPtr[455]=-314.0731361794753;
    cooRowIndexHostPtr[456]=114; cooColIndexHostPtr[456]=115; cooValHostPtr[456]=155.0365680897376;
    cooRowIndexHostPtr[457]=115; cooColIndexHostPtr[457]=15; cooValHostPtr[457]=-5.45;
    cooRowIndexHostPtr[458]=115; cooColIndexHostPtr[458]=114; cooValHostPtr[458]=155.0365680897376;
    cooRowIndexHostPtr[459]=115; cooColIndexHostPtr[459]=115; cooValHostPtr[459]=-314.0731361794753;
    cooRowIndexHostPtr[460]=115; cooColIndexHostPtr[460]=116; cooValHostPtr[460]=155.0365680897376;
    cooRowIndexHostPtr[461]=116; cooColIndexHostPtr[461]=16; cooValHostPtr[461]=-5.45;
    cooRowIndexHostPtr[462]=116; cooColIndexHostPtr[462]=115; cooValHostPtr[462]=155.0365680897376;
    cooRowIndexHostPtr[463]=116; cooColIndexHostPtr[463]=116; cooValHostPtr[463]=-314.0731361794753;
    cooRowIndexHostPtr[464]=116; cooColIndexHostPtr[464]=117; cooValHostPtr[464]=155.0365680897376;
    cooRowIndexHostPtr[465]=117; cooColIndexHostPtr[465]=17; cooValHostPtr[465]=-5.45;
    cooRowIndexHostPtr[466]=117; cooColIndexHostPtr[466]=116; cooValHostPtr[466]=155.0365680897376;
    cooRowIndexHostPtr[467]=117; cooColIndexHostPtr[467]=117; cooValHostPtr[467]=-314.0731361794753;
    cooRowIndexHostPtr[468]=117; cooColIndexHostPtr[468]=118; cooValHostPtr[468]=155.0365680897376;
    cooRowIndexHostPtr[469]=118; cooColIndexHostPtr[469]=18; cooValHostPtr[469]=-5.45;
    cooRowIndexHostPtr[470]=118; cooColIndexHostPtr[470]=117; cooValHostPtr[470]=155.0365680897376;
    cooRowIndexHostPtr[471]=118; cooColIndexHostPtr[471]=118; cooValHostPtr[471]=-314.0731361794753;
    cooRowIndexHostPtr[472]=118; cooColIndexHostPtr[472]=119; cooValHostPtr[472]=155.0365680897376;
    cooRowIndexHostPtr[473]=119; cooColIndexHostPtr[473]=19; cooValHostPtr[473]=-5.45;
    cooRowIndexHostPtr[474]=119; cooColIndexHostPtr[474]=118; cooValHostPtr[474]=155.0365680897376;
    cooRowIndexHostPtr[475]=119; cooColIndexHostPtr[475]=119; cooValHostPtr[475]=-314.0731361794753;
    cooRowIndexHostPtr[476]=119; cooColIndexHostPtr[476]=120; cooValHostPtr[476]=155.0365680897376;
    cooRowIndexHostPtr[477]=120; cooColIndexHostPtr[477]=20; cooValHostPtr[477]=-5.45;
    cooRowIndexHostPtr[478]=120; cooColIndexHostPtr[478]=119; cooValHostPtr[478]=155.0365680897376;
    cooRowIndexHostPtr[479]=120; cooColIndexHostPtr[479]=120; cooValHostPtr[479]=-314.0731361794753;
    cooRowIndexHostPtr[480]=120; cooColIndexHostPtr[480]=121; cooValHostPtr[480]=155.0365680897376;
    cooRowIndexHostPtr[481]=121; cooColIndexHostPtr[481]=21; cooValHostPtr[481]=-5.45;
    cooRowIndexHostPtr[482]=121; cooColIndexHostPtr[482]=120; cooValHostPtr[482]=155.0365680897376;
    cooRowIndexHostPtr[483]=121; cooColIndexHostPtr[483]=121; cooValHostPtr[483]=-314.0731361794753;
    cooRowIndexHostPtr[484]=121; cooColIndexHostPtr[484]=122; cooValHostPtr[484]=155.0365680897376;
    cooRowIndexHostPtr[485]=122; cooColIndexHostPtr[485]=22; cooValHostPtr[485]=-5.45;
    cooRowIndexHostPtr[486]=122; cooColIndexHostPtr[486]=121; cooValHostPtr[486]=155.0365680897376;
    cooRowIndexHostPtr[487]=122; cooColIndexHostPtr[487]=122; cooValHostPtr[487]=-314.0731361794753;
    cooRowIndexHostPtr[488]=122; cooColIndexHostPtr[488]=123; cooValHostPtr[488]=155.0365680897376;
    cooRowIndexHostPtr[489]=123; cooColIndexHostPtr[489]=23; cooValHostPtr[489]=-5.45;
    cooRowIndexHostPtr[490]=123; cooColIndexHostPtr[490]=122; cooValHostPtr[490]=155.0365680897376;
    cooRowIndexHostPtr[491]=123; cooColIndexHostPtr[491]=123; cooValHostPtr[491]=-314.0731361794753;
    cooRowIndexHostPtr[492]=123; cooColIndexHostPtr[492]=124; cooValHostPtr[492]=155.0365680897376;
    cooRowIndexHostPtr[493]=124; cooColIndexHostPtr[493]=24; cooValHostPtr[493]=-5.45;
    cooRowIndexHostPtr[494]=124; cooColIndexHostPtr[494]=123; cooValHostPtr[494]=155.0365680897376;
    cooRowIndexHostPtr[495]=124; cooColIndexHostPtr[495]=124; cooValHostPtr[495]=-314.0731361794753;
    cooRowIndexHostPtr[496]=124; cooColIndexHostPtr[496]=125; cooValHostPtr[496]=155.0365680897376;
    cooRowIndexHostPtr[497]=125; cooColIndexHostPtr[497]=25; cooValHostPtr[497]=-5.45;
    cooRowIndexHostPtr[498]=125; cooColIndexHostPtr[498]=124; cooValHostPtr[498]=155.0365680897376;
    cooRowIndexHostPtr[499]=125; cooColIndexHostPtr[499]=125; cooValHostPtr[499]=-314.0731361794753;
    cooRowIndexHostPtr[500]=125; cooColIndexHostPtr[500]=126; cooValHostPtr[500]=155.0365680897376;
    cooRowIndexHostPtr[501]=126; cooColIndexHostPtr[501]=26; cooValHostPtr[501]=-5.45;
    cooRowIndexHostPtr[502]=126; cooColIndexHostPtr[502]=125; cooValHostPtr[502]=155.0365680897376;
    cooRowIndexHostPtr[503]=126; cooColIndexHostPtr[503]=126; cooValHostPtr[503]=-314.0731361794753;
    cooRowIndexHostPtr[504]=126; cooColIndexHostPtr[504]=127; cooValHostPtr[504]=155.0365680897376;
    cooRowIndexHostPtr[505]=127; cooColIndexHostPtr[505]=27; cooValHostPtr[505]=-5.45;
    cooRowIndexHostPtr[506]=127; cooColIndexHostPtr[506]=126; cooValHostPtr[506]=155.0365680897376;
    cooRowIndexHostPtr[507]=127; cooColIndexHostPtr[507]=127; cooValHostPtr[507]=-314.0731361794753;
    cooRowIndexHostPtr[508]=127; cooColIndexHostPtr[508]=128; cooValHostPtr[508]=155.0365680897376;
    cooRowIndexHostPtr[509]=128; cooColIndexHostPtr[509]=28; cooValHostPtr[509]=-5.45;
    cooRowIndexHostPtr[510]=128; cooColIndexHostPtr[510]=127; cooValHostPtr[510]=155.0365680897376;
    cooRowIndexHostPtr[511]=128; cooColIndexHostPtr[511]=128; cooValHostPtr[511]=-314.0731361794753;
    cooRowIndexHostPtr[512]=128; cooColIndexHostPtr[512]=129; cooValHostPtr[512]=155.0365680897376;
    cooRowIndexHostPtr[513]=129; cooColIndexHostPtr[513]=29; cooValHostPtr[513]=-5.45;
    cooRowIndexHostPtr[514]=129; cooColIndexHostPtr[514]=128; cooValHostPtr[514]=155.0365680897376;
    cooRowIndexHostPtr[515]=129; cooColIndexHostPtr[515]=129; cooValHostPtr[515]=-314.0731361794753;
    cooRowIndexHostPtr[516]=129; cooColIndexHostPtr[516]=130; cooValHostPtr[516]=155.0365680897376;
    cooRowIndexHostPtr[517]=130; cooColIndexHostPtr[517]=30; cooValHostPtr[517]=-5.45;
    cooRowIndexHostPtr[518]=130; cooColIndexHostPtr[518]=129; cooValHostPtr[518]=155.0365680897376;
    cooRowIndexHostPtr[519]=130; cooColIndexHostPtr[519]=130; cooValHostPtr[519]=-314.0731361794753;
    cooRowIndexHostPtr[520]=130; cooColIndexHostPtr[520]=131; cooValHostPtr[520]=155.0365680897376;
    cooRowIndexHostPtr[521]=131; cooColIndexHostPtr[521]=31; cooValHostPtr[521]=-5.45;
    cooRowIndexHostPtr[522]=131; cooColIndexHostPtr[522]=130; cooValHostPtr[522]=155.0365680897376;
    cooRowIndexHostPtr[523]=131; cooColIndexHostPtr[523]=131; cooValHostPtr[523]=-314.0731361794753;
    cooRowIndexHostPtr[524]=131; cooColIndexHostPtr[524]=132; cooValHostPtr[524]=155.0365680897376;
    cooRowIndexHostPtr[525]=132; cooColIndexHostPtr[525]=32; cooValHostPtr[525]=-5.45;
    cooRowIndexHostPtr[526]=132; cooColIndexHostPtr[526]=131; cooValHostPtr[526]=155.0365680897376;
    cooRowIndexHostPtr[527]=132; cooColIndexHostPtr[527]=132; cooValHostPtr[527]=-314.0731361794753;
    cooRowIndexHostPtr[528]=132; cooColIndexHostPtr[528]=133; cooValHostPtr[528]=155.0365680897376;
    cooRowIndexHostPtr[529]=133; cooColIndexHostPtr[529]=33; cooValHostPtr[529]=-5.45;
    cooRowIndexHostPtr[530]=133; cooColIndexHostPtr[530]=132; cooValHostPtr[530]=155.0365680897376;
    cooRowIndexHostPtr[531]=133; cooColIndexHostPtr[531]=133; cooValHostPtr[531]=-314.0731361794753;
    cooRowIndexHostPtr[532]=133; cooColIndexHostPtr[532]=134; cooValHostPtr[532]=155.0365680897376;
    cooRowIndexHostPtr[533]=134; cooColIndexHostPtr[533]=34; cooValHostPtr[533]=-5.45;
    cooRowIndexHostPtr[534]=134; cooColIndexHostPtr[534]=133; cooValHostPtr[534]=155.0365680897376;
    cooRowIndexHostPtr[535]=134; cooColIndexHostPtr[535]=134; cooValHostPtr[535]=-314.0731361794753;
    cooRowIndexHostPtr[536]=134; cooColIndexHostPtr[536]=135; cooValHostPtr[536]=155.0365680897376;
    cooRowIndexHostPtr[537]=135; cooColIndexHostPtr[537]=35; cooValHostPtr[537]=-5.45;
    cooRowIndexHostPtr[538]=135; cooColIndexHostPtr[538]=134; cooValHostPtr[538]=155.0365680897376;
    cooRowIndexHostPtr[539]=135; cooColIndexHostPtr[539]=135; cooValHostPtr[539]=-314.0731361794753;
    cooRowIndexHostPtr[540]=135; cooColIndexHostPtr[540]=136; cooValHostPtr[540]=155.0365680897376;
    cooRowIndexHostPtr[541]=136; cooColIndexHostPtr[541]=36; cooValHostPtr[541]=-5.45;
    cooRowIndexHostPtr[542]=136; cooColIndexHostPtr[542]=135; cooValHostPtr[542]=155.0365680897376;
    cooRowIndexHostPtr[543]=136; cooColIndexHostPtr[543]=136; cooValHostPtr[543]=-314.0731361794753;
    cooRowIndexHostPtr[544]=136; cooColIndexHostPtr[544]=137; cooValHostPtr[544]=155.0365680897376;
    cooRowIndexHostPtr[545]=137; cooColIndexHostPtr[545]=37; cooValHostPtr[545]=-5.45;
    cooRowIndexHostPtr[546]=137; cooColIndexHostPtr[546]=136; cooValHostPtr[546]=155.0365680897376;
    cooRowIndexHostPtr[547]=137; cooColIndexHostPtr[547]=137; cooValHostPtr[547]=-314.0731361794753;
    cooRowIndexHostPtr[548]=137; cooColIndexHostPtr[548]=138; cooValHostPtr[548]=155.0365680897376;
    cooRowIndexHostPtr[549]=138; cooColIndexHostPtr[549]=38; cooValHostPtr[549]=-5.45;
    cooRowIndexHostPtr[550]=138; cooColIndexHostPtr[550]=137; cooValHostPtr[550]=155.0365680897376;
    cooRowIndexHostPtr[551]=138; cooColIndexHostPtr[551]=138; cooValHostPtr[551]=-314.0731361794753;
    cooRowIndexHostPtr[552]=138; cooColIndexHostPtr[552]=139; cooValHostPtr[552]=155.0365680897376;
    cooRowIndexHostPtr[553]=139; cooColIndexHostPtr[553]=39; cooValHostPtr[553]=-5.45;
    cooRowIndexHostPtr[554]=139; cooColIndexHostPtr[554]=138; cooValHostPtr[554]=155.0365680897376;
    cooRowIndexHostPtr[555]=139; cooColIndexHostPtr[555]=139; cooValHostPtr[555]=-314.0731361794753;
    cooRowIndexHostPtr[556]=139; cooColIndexHostPtr[556]=140; cooValHostPtr[556]=155.0365680897376;
    cooRowIndexHostPtr[557]=140; cooColIndexHostPtr[557]=40; cooValHostPtr[557]=-5.45;
    cooRowIndexHostPtr[558]=140; cooColIndexHostPtr[558]=139; cooValHostPtr[558]=155.0365680897376;
    cooRowIndexHostPtr[559]=140; cooColIndexHostPtr[559]=140; cooValHostPtr[559]=-314.0731361794753;
    cooRowIndexHostPtr[560]=140; cooColIndexHostPtr[560]=141; cooValHostPtr[560]=155.0365680897376;
    cooRowIndexHostPtr[561]=141; cooColIndexHostPtr[561]=41; cooValHostPtr[561]=-5.45;
    cooRowIndexHostPtr[562]=141; cooColIndexHostPtr[562]=140; cooValHostPtr[562]=155.0365680897376;
    cooRowIndexHostPtr[563]=141; cooColIndexHostPtr[563]=141; cooValHostPtr[563]=-314.0731361794753;
    cooRowIndexHostPtr[564]=141; cooColIndexHostPtr[564]=142; cooValHostPtr[564]=155.0365680897376;
    cooRowIndexHostPtr[565]=142; cooColIndexHostPtr[565]=42; cooValHostPtr[565]=-5.45;
    cooRowIndexHostPtr[566]=142; cooColIndexHostPtr[566]=141; cooValHostPtr[566]=155.0365680897376;
    cooRowIndexHostPtr[567]=142; cooColIndexHostPtr[567]=142; cooValHostPtr[567]=-314.0731361794753;
    cooRowIndexHostPtr[568]=142; cooColIndexHostPtr[568]=143; cooValHostPtr[568]=155.0365680897376;
    cooRowIndexHostPtr[569]=143; cooColIndexHostPtr[569]=43; cooValHostPtr[569]=-5.45;
    cooRowIndexHostPtr[570]=143; cooColIndexHostPtr[570]=142; cooValHostPtr[570]=155.0365680897376;
    cooRowIndexHostPtr[571]=143; cooColIndexHostPtr[571]=143; cooValHostPtr[571]=-314.0731361794753;
    cooRowIndexHostPtr[572]=143; cooColIndexHostPtr[572]=144; cooValHostPtr[572]=155.0365680897376;
    cooRowIndexHostPtr[573]=144; cooColIndexHostPtr[573]=44; cooValHostPtr[573]=-5.45;
    cooRowIndexHostPtr[574]=144; cooColIndexHostPtr[574]=143; cooValHostPtr[574]=155.0365680897376;
    cooRowIndexHostPtr[575]=144; cooColIndexHostPtr[575]=144; cooValHostPtr[575]=-314.0731361794753;
    cooRowIndexHostPtr[576]=144; cooColIndexHostPtr[576]=145; cooValHostPtr[576]=155.0365680897376;
    cooRowIndexHostPtr[577]=145; cooColIndexHostPtr[577]=45; cooValHostPtr[577]=-5.45;
    cooRowIndexHostPtr[578]=145; cooColIndexHostPtr[578]=144; cooValHostPtr[578]=155.0365680897376;
    cooRowIndexHostPtr[579]=145; cooColIndexHostPtr[579]=145; cooValHostPtr[579]=-314.0731361794753;
    cooRowIndexHostPtr[580]=145; cooColIndexHostPtr[580]=146; cooValHostPtr[580]=155.0365680897376;
    cooRowIndexHostPtr[581]=146; cooColIndexHostPtr[581]=46; cooValHostPtr[581]=-5.45;
    cooRowIndexHostPtr[582]=146; cooColIndexHostPtr[582]=145; cooValHostPtr[582]=155.0365680897376;
    cooRowIndexHostPtr[583]=146; cooColIndexHostPtr[583]=146; cooValHostPtr[583]=-314.0731361794753;
    cooRowIndexHostPtr[584]=146; cooColIndexHostPtr[584]=147; cooValHostPtr[584]=155.0365680897376;
    cooRowIndexHostPtr[585]=147; cooColIndexHostPtr[585]=47; cooValHostPtr[585]=-5.45;
    cooRowIndexHostPtr[586]=147; cooColIndexHostPtr[586]=146; cooValHostPtr[586]=155.0365680897376;
    cooRowIndexHostPtr[587]=147; cooColIndexHostPtr[587]=147; cooValHostPtr[587]=-314.0731361794753;
    cooRowIndexHostPtr[588]=147; cooColIndexHostPtr[588]=148; cooValHostPtr[588]=155.0365680897376;
    cooRowIndexHostPtr[589]=148; cooColIndexHostPtr[589]=48; cooValHostPtr[589]=-5.45;
    cooRowIndexHostPtr[590]=148; cooColIndexHostPtr[590]=147; cooValHostPtr[590]=155.0365680897376;
    cooRowIndexHostPtr[591]=148; cooColIndexHostPtr[591]=148; cooValHostPtr[591]=-314.0731361794753;
    cooRowIndexHostPtr[592]=148; cooColIndexHostPtr[592]=149; cooValHostPtr[592]=155.0365680897376;
    cooRowIndexHostPtr[593]=149; cooColIndexHostPtr[593]=49; cooValHostPtr[593]=-5.45;
    cooRowIndexHostPtr[594]=149; cooColIndexHostPtr[594]=148; cooValHostPtr[594]=155.0365680897376;
    cooRowIndexHostPtr[595]=149; cooColIndexHostPtr[595]=149; cooValHostPtr[595]=-314.0731361794753;
    cooRowIndexHostPtr[596]=149; cooColIndexHostPtr[596]=150; cooValHostPtr[596]=155.0365680897376;
    cooRowIndexHostPtr[597]=150; cooColIndexHostPtr[597]=50; cooValHostPtr[597]=-5.45;
    cooRowIndexHostPtr[598]=150; cooColIndexHostPtr[598]=149; cooValHostPtr[598]=155.0365680897376;
    cooRowIndexHostPtr[599]=150; cooColIndexHostPtr[599]=150; cooValHostPtr[599]=-314.0731361794753;
    cooRowIndexHostPtr[600]=150; cooColIndexHostPtr[600]=151; cooValHostPtr[600]=155.0365680897376;
    cooRowIndexHostPtr[601]=151; cooColIndexHostPtr[601]=51; cooValHostPtr[601]=-5.45;
    cooRowIndexHostPtr[602]=151; cooColIndexHostPtr[602]=150; cooValHostPtr[602]=155.0365680897376;
    cooRowIndexHostPtr[603]=151; cooColIndexHostPtr[603]=151; cooValHostPtr[603]=-314.0731361794753;
    cooRowIndexHostPtr[604]=151; cooColIndexHostPtr[604]=152; cooValHostPtr[604]=155.0365680897376;
    cooRowIndexHostPtr[605]=152; cooColIndexHostPtr[605]=52; cooValHostPtr[605]=-5.45;
    cooRowIndexHostPtr[606]=152; cooColIndexHostPtr[606]=151; cooValHostPtr[606]=155.0365680897376;
    cooRowIndexHostPtr[607]=152; cooColIndexHostPtr[607]=152; cooValHostPtr[607]=-314.0731361794753;
    cooRowIndexHostPtr[608]=152; cooColIndexHostPtr[608]=153; cooValHostPtr[608]=155.0365680897376;
    cooRowIndexHostPtr[609]=153; cooColIndexHostPtr[609]=53; cooValHostPtr[609]=-5.45;
    cooRowIndexHostPtr[610]=153; cooColIndexHostPtr[610]=152; cooValHostPtr[610]=155.0365680897376;
    cooRowIndexHostPtr[611]=153; cooColIndexHostPtr[611]=153; cooValHostPtr[611]=-314.0731361794753;
    cooRowIndexHostPtr[612]=153; cooColIndexHostPtr[612]=154; cooValHostPtr[612]=155.0365680897376;
    cooRowIndexHostPtr[613]=154; cooColIndexHostPtr[613]=54; cooValHostPtr[613]=-5.45;
    cooRowIndexHostPtr[614]=154; cooColIndexHostPtr[614]=153; cooValHostPtr[614]=155.0365680897376;
    cooRowIndexHostPtr[615]=154; cooColIndexHostPtr[615]=154; cooValHostPtr[615]=-314.0731361794753;
    cooRowIndexHostPtr[616]=154; cooColIndexHostPtr[616]=155; cooValHostPtr[616]=155.0365680897376;
    cooRowIndexHostPtr[617]=155; cooColIndexHostPtr[617]=55; cooValHostPtr[617]=-5.45;
    cooRowIndexHostPtr[618]=155; cooColIndexHostPtr[618]=154; cooValHostPtr[618]=155.0365680897376;
    cooRowIndexHostPtr[619]=155; cooColIndexHostPtr[619]=155; cooValHostPtr[619]=-314.0731361794753;
    cooRowIndexHostPtr[620]=155; cooColIndexHostPtr[620]=156; cooValHostPtr[620]=155.0365680897376;
    cooRowIndexHostPtr[621]=156; cooColIndexHostPtr[621]=56; cooValHostPtr[621]=-5.45;
    cooRowIndexHostPtr[622]=156; cooColIndexHostPtr[622]=155; cooValHostPtr[622]=155.0365680897376;
    cooRowIndexHostPtr[623]=156; cooColIndexHostPtr[623]=156; cooValHostPtr[623]=-314.0731361794753;
    cooRowIndexHostPtr[624]=156; cooColIndexHostPtr[624]=157; cooValHostPtr[624]=155.0365680897376;
    cooRowIndexHostPtr[625]=157; cooColIndexHostPtr[625]=57; cooValHostPtr[625]=-5.45;
    cooRowIndexHostPtr[626]=157; cooColIndexHostPtr[626]=156; cooValHostPtr[626]=155.0365680897376;
    cooRowIndexHostPtr[627]=157; cooColIndexHostPtr[627]=157; cooValHostPtr[627]=-314.0731361794753;
    cooRowIndexHostPtr[628]=157; cooColIndexHostPtr[628]=158; cooValHostPtr[628]=155.0365680897376;
    cooRowIndexHostPtr[629]=158; cooColIndexHostPtr[629]=58; cooValHostPtr[629]=-5.45;
    cooRowIndexHostPtr[630]=158; cooColIndexHostPtr[630]=157; cooValHostPtr[630]=155.0365680897376;
    cooRowIndexHostPtr[631]=158; cooColIndexHostPtr[631]=158; cooValHostPtr[631]=-314.0731361794753;
    cooRowIndexHostPtr[632]=158; cooColIndexHostPtr[632]=159; cooValHostPtr[632]=155.0365680897376;
    cooRowIndexHostPtr[633]=159; cooColIndexHostPtr[633]=59; cooValHostPtr[633]=-5.45;
    cooRowIndexHostPtr[634]=159; cooColIndexHostPtr[634]=158; cooValHostPtr[634]=155.0365680897376;
    cooRowIndexHostPtr[635]=159; cooColIndexHostPtr[635]=159; cooValHostPtr[635]=-314.0731361794753;
    cooRowIndexHostPtr[636]=159; cooColIndexHostPtr[636]=160; cooValHostPtr[636]=155.0365680897376;
    cooRowIndexHostPtr[637]=160; cooColIndexHostPtr[637]=60; cooValHostPtr[637]=-5.45;
    cooRowIndexHostPtr[638]=160; cooColIndexHostPtr[638]=159; cooValHostPtr[638]=155.0365680897376;
    cooRowIndexHostPtr[639]=160; cooColIndexHostPtr[639]=160; cooValHostPtr[639]=-314.0731361794753;
    cooRowIndexHostPtr[640]=160; cooColIndexHostPtr[640]=161; cooValHostPtr[640]=155.0365680897376;
    cooRowIndexHostPtr[641]=161; cooColIndexHostPtr[641]=61; cooValHostPtr[641]=-5.45;
    cooRowIndexHostPtr[642]=161; cooColIndexHostPtr[642]=160; cooValHostPtr[642]=155.0365680897376;
    cooRowIndexHostPtr[643]=161; cooColIndexHostPtr[643]=161; cooValHostPtr[643]=-314.0731361794753;
    cooRowIndexHostPtr[644]=161; cooColIndexHostPtr[644]=162; cooValHostPtr[644]=155.0365680897376;
    cooRowIndexHostPtr[645]=162; cooColIndexHostPtr[645]=62; cooValHostPtr[645]=-5.45;
    cooRowIndexHostPtr[646]=162; cooColIndexHostPtr[646]=161; cooValHostPtr[646]=155.0365680897376;
    cooRowIndexHostPtr[647]=162; cooColIndexHostPtr[647]=162; cooValHostPtr[647]=-314.0731361794753;
    cooRowIndexHostPtr[648]=162; cooColIndexHostPtr[648]=163; cooValHostPtr[648]=155.0365680897376;
    cooRowIndexHostPtr[649]=163; cooColIndexHostPtr[649]=63; cooValHostPtr[649]=-5.45;
    cooRowIndexHostPtr[650]=163; cooColIndexHostPtr[650]=162; cooValHostPtr[650]=155.0365680897376;
    cooRowIndexHostPtr[651]=163; cooColIndexHostPtr[651]=163; cooValHostPtr[651]=-314.0731361794753;
    cooRowIndexHostPtr[652]=163; cooColIndexHostPtr[652]=164; cooValHostPtr[652]=155.0365680897376;
    cooRowIndexHostPtr[653]=164; cooColIndexHostPtr[653]=64; cooValHostPtr[653]=-5.45;
    cooRowIndexHostPtr[654]=164; cooColIndexHostPtr[654]=163; cooValHostPtr[654]=155.0365680897376;
    cooRowIndexHostPtr[655]=164; cooColIndexHostPtr[655]=164; cooValHostPtr[655]=-314.0731361794753;
    cooRowIndexHostPtr[656]=164; cooColIndexHostPtr[656]=165; cooValHostPtr[656]=155.0365680897376;
    cooRowIndexHostPtr[657]=165; cooColIndexHostPtr[657]=65; cooValHostPtr[657]=-5.45;
    cooRowIndexHostPtr[658]=165; cooColIndexHostPtr[658]=164; cooValHostPtr[658]=155.0365680897376;
    cooRowIndexHostPtr[659]=165; cooColIndexHostPtr[659]=165; cooValHostPtr[659]=-314.0731361794753;
    cooRowIndexHostPtr[660]=165; cooColIndexHostPtr[660]=166; cooValHostPtr[660]=155.0365680897376;
    cooRowIndexHostPtr[661]=166; cooColIndexHostPtr[661]=66; cooValHostPtr[661]=-5.45;
    cooRowIndexHostPtr[662]=166; cooColIndexHostPtr[662]=165; cooValHostPtr[662]=155.0365680897376;
    cooRowIndexHostPtr[663]=166; cooColIndexHostPtr[663]=166; cooValHostPtr[663]=-314.0731361794753;
    cooRowIndexHostPtr[664]=166; cooColIndexHostPtr[664]=167; cooValHostPtr[664]=155.0365680897376;
    cooRowIndexHostPtr[665]=167; cooColIndexHostPtr[665]=67; cooValHostPtr[665]=-5.45;
    cooRowIndexHostPtr[666]=167; cooColIndexHostPtr[666]=166; cooValHostPtr[666]=155.0365680897376;
    cooRowIndexHostPtr[667]=167; cooColIndexHostPtr[667]=167; cooValHostPtr[667]=-314.0731361794753;
    cooRowIndexHostPtr[668]=167; cooColIndexHostPtr[668]=168; cooValHostPtr[668]=155.0365680897376;
    cooRowIndexHostPtr[669]=168; cooColIndexHostPtr[669]=68; cooValHostPtr[669]=-5.45;
    cooRowIndexHostPtr[670]=168; cooColIndexHostPtr[670]=167; cooValHostPtr[670]=155.0365680897376;
    cooRowIndexHostPtr[671]=168; cooColIndexHostPtr[671]=168; cooValHostPtr[671]=-314.0731361794753;
    cooRowIndexHostPtr[672]=168; cooColIndexHostPtr[672]=169; cooValHostPtr[672]=155.0365680897376;
    cooRowIndexHostPtr[673]=169; cooColIndexHostPtr[673]=69; cooValHostPtr[673]=-5.45;
    cooRowIndexHostPtr[674]=169; cooColIndexHostPtr[674]=168; cooValHostPtr[674]=155.0365680897376;
    cooRowIndexHostPtr[675]=169; cooColIndexHostPtr[675]=169; cooValHostPtr[675]=-314.0731361794753;
    cooRowIndexHostPtr[676]=169; cooColIndexHostPtr[676]=170; cooValHostPtr[676]=155.0365680897376;
    cooRowIndexHostPtr[677]=170; cooColIndexHostPtr[677]=70; cooValHostPtr[677]=-5.45;
    cooRowIndexHostPtr[678]=170; cooColIndexHostPtr[678]=169; cooValHostPtr[678]=155.0365680897376;
    cooRowIndexHostPtr[679]=170; cooColIndexHostPtr[679]=170; cooValHostPtr[679]=-314.0731361794753;
    cooRowIndexHostPtr[680]=170; cooColIndexHostPtr[680]=171; cooValHostPtr[680]=155.0365680897376;
    cooRowIndexHostPtr[681]=171; cooColIndexHostPtr[681]=71; cooValHostPtr[681]=-5.45;
    cooRowIndexHostPtr[682]=171; cooColIndexHostPtr[682]=170; cooValHostPtr[682]=155.0365680897376;
    cooRowIndexHostPtr[683]=171; cooColIndexHostPtr[683]=171; cooValHostPtr[683]=-314.0731361794753;
    cooRowIndexHostPtr[684]=171; cooColIndexHostPtr[684]=172; cooValHostPtr[684]=155.0365680897376;
    cooRowIndexHostPtr[685]=172; cooColIndexHostPtr[685]=72; cooValHostPtr[685]=-5.45;
    cooRowIndexHostPtr[686]=172; cooColIndexHostPtr[686]=171; cooValHostPtr[686]=155.0365680897376;
    cooRowIndexHostPtr[687]=172; cooColIndexHostPtr[687]=172; cooValHostPtr[687]=-314.0731361794753;
    cooRowIndexHostPtr[688]=172; cooColIndexHostPtr[688]=173; cooValHostPtr[688]=155.0365680897376;
    cooRowIndexHostPtr[689]=173; cooColIndexHostPtr[689]=73; cooValHostPtr[689]=-5.45;
    cooRowIndexHostPtr[690]=173; cooColIndexHostPtr[690]=172; cooValHostPtr[690]=155.0365680897376;
    cooRowIndexHostPtr[691]=173; cooColIndexHostPtr[691]=173; cooValHostPtr[691]=-314.0731361794753;
    cooRowIndexHostPtr[692]=173; cooColIndexHostPtr[692]=174; cooValHostPtr[692]=155.0365680897376;
    cooRowIndexHostPtr[693]=174; cooColIndexHostPtr[693]=74; cooValHostPtr[693]=-5.45;
    cooRowIndexHostPtr[694]=174; cooColIndexHostPtr[694]=173; cooValHostPtr[694]=155.0365680897376;
    cooRowIndexHostPtr[695]=174; cooColIndexHostPtr[695]=174; cooValHostPtr[695]=-314.0731361794753;
    cooRowIndexHostPtr[696]=174; cooColIndexHostPtr[696]=175; cooValHostPtr[696]=155.0365680897376;
    cooRowIndexHostPtr[697]=175; cooColIndexHostPtr[697]=75; cooValHostPtr[697]=-5.45;
    cooRowIndexHostPtr[698]=175; cooColIndexHostPtr[698]=174; cooValHostPtr[698]=155.0365680897376;
    cooRowIndexHostPtr[699]=175; cooColIndexHostPtr[699]=175; cooValHostPtr[699]=-314.0731361794753;
    cooRowIndexHostPtr[700]=175; cooColIndexHostPtr[700]=176; cooValHostPtr[700]=155.0365680897376;
    cooRowIndexHostPtr[701]=176; cooColIndexHostPtr[701]=76; cooValHostPtr[701]=-5.45;
    cooRowIndexHostPtr[702]=176; cooColIndexHostPtr[702]=175; cooValHostPtr[702]=155.0365680897376;
    cooRowIndexHostPtr[703]=176; cooColIndexHostPtr[703]=176; cooValHostPtr[703]=-314.0731361794753;
    cooRowIndexHostPtr[704]=176; cooColIndexHostPtr[704]=177; cooValHostPtr[704]=155.0365680897376;
    cooRowIndexHostPtr[705]=177; cooColIndexHostPtr[705]=77; cooValHostPtr[705]=-5.45;
    cooRowIndexHostPtr[706]=177; cooColIndexHostPtr[706]=176; cooValHostPtr[706]=155.0365680897376;
    cooRowIndexHostPtr[707]=177; cooColIndexHostPtr[707]=177; cooValHostPtr[707]=-314.0731361794753;
    cooRowIndexHostPtr[708]=177; cooColIndexHostPtr[708]=178; cooValHostPtr[708]=155.0365680897376;
    cooRowIndexHostPtr[709]=178; cooColIndexHostPtr[709]=78; cooValHostPtr[709]=-5.45;
    cooRowIndexHostPtr[710]=178; cooColIndexHostPtr[710]=177; cooValHostPtr[710]=155.0365680897376;
    cooRowIndexHostPtr[711]=178; cooColIndexHostPtr[711]=178; cooValHostPtr[711]=-314.0731361794753;
    cooRowIndexHostPtr[712]=178; cooColIndexHostPtr[712]=179; cooValHostPtr[712]=155.0365680897376;
    cooRowIndexHostPtr[713]=179; cooColIndexHostPtr[713]=79; cooValHostPtr[713]=-5.45;
    cooRowIndexHostPtr[714]=179; cooColIndexHostPtr[714]=178; cooValHostPtr[714]=155.0365680897376;
    cooRowIndexHostPtr[715]=179; cooColIndexHostPtr[715]=179; cooValHostPtr[715]=-314.0731361794753;
    cooRowIndexHostPtr[716]=179; cooColIndexHostPtr[716]=180; cooValHostPtr[716]=155.0365680897376;
    cooRowIndexHostPtr[717]=180; cooColIndexHostPtr[717]=80; cooValHostPtr[717]=-5.45;
    cooRowIndexHostPtr[718]=180; cooColIndexHostPtr[718]=179; cooValHostPtr[718]=155.0365680897376;
    cooRowIndexHostPtr[719]=180; cooColIndexHostPtr[719]=180; cooValHostPtr[719]=-314.0731361794753;
    cooRowIndexHostPtr[720]=180; cooColIndexHostPtr[720]=181; cooValHostPtr[720]=155.0365680897376;
    cooRowIndexHostPtr[721]=181; cooColIndexHostPtr[721]=81; cooValHostPtr[721]=-5.45;
    cooRowIndexHostPtr[722]=181; cooColIndexHostPtr[722]=180; cooValHostPtr[722]=155.0365680897376;
    cooRowIndexHostPtr[723]=181; cooColIndexHostPtr[723]=181; cooValHostPtr[723]=-314.0731361794753;
    cooRowIndexHostPtr[724]=181; cooColIndexHostPtr[724]=182; cooValHostPtr[724]=155.0365680897376;
    cooRowIndexHostPtr[725]=182; cooColIndexHostPtr[725]=82; cooValHostPtr[725]=-5.45;
    cooRowIndexHostPtr[726]=182; cooColIndexHostPtr[726]=181; cooValHostPtr[726]=155.0365680897376;
    cooRowIndexHostPtr[727]=182; cooColIndexHostPtr[727]=182; cooValHostPtr[727]=-314.0731361794753;
    cooRowIndexHostPtr[728]=182; cooColIndexHostPtr[728]=183; cooValHostPtr[728]=155.0365680897376;
    cooRowIndexHostPtr[729]=183; cooColIndexHostPtr[729]=83; cooValHostPtr[729]=-5.45;
    cooRowIndexHostPtr[730]=183; cooColIndexHostPtr[730]=182; cooValHostPtr[730]=155.0365680897376;
    cooRowIndexHostPtr[731]=183; cooColIndexHostPtr[731]=183; cooValHostPtr[731]=-314.0731361794753;
    cooRowIndexHostPtr[732]=183; cooColIndexHostPtr[732]=184; cooValHostPtr[732]=155.0365680897376;
    cooRowIndexHostPtr[733]=184; cooColIndexHostPtr[733]=84; cooValHostPtr[733]=-5.45;
    cooRowIndexHostPtr[734]=184; cooColIndexHostPtr[734]=183; cooValHostPtr[734]=155.0365680897376;
    cooRowIndexHostPtr[735]=184; cooColIndexHostPtr[735]=184; cooValHostPtr[735]=-314.0731361794753;
    cooRowIndexHostPtr[736]=184; cooColIndexHostPtr[736]=185; cooValHostPtr[736]=155.0365680897376;
    cooRowIndexHostPtr[737]=185; cooColIndexHostPtr[737]=85; cooValHostPtr[737]=-5.45;
    cooRowIndexHostPtr[738]=185; cooColIndexHostPtr[738]=184; cooValHostPtr[738]=155.0365680897376;
    cooRowIndexHostPtr[739]=185; cooColIndexHostPtr[739]=185; cooValHostPtr[739]=-314.0731361794753;
    cooRowIndexHostPtr[740]=185; cooColIndexHostPtr[740]=186; cooValHostPtr[740]=155.0365680897376;
    cooRowIndexHostPtr[741]=186; cooColIndexHostPtr[741]=86; cooValHostPtr[741]=-5.45;
    cooRowIndexHostPtr[742]=186; cooColIndexHostPtr[742]=185; cooValHostPtr[742]=155.0365680897376;
    cooRowIndexHostPtr[743]=186; cooColIndexHostPtr[743]=186; cooValHostPtr[743]=-314.0731361794753;
    cooRowIndexHostPtr[744]=186; cooColIndexHostPtr[744]=187; cooValHostPtr[744]=155.0365680897376;
    cooRowIndexHostPtr[745]=187; cooColIndexHostPtr[745]=87; cooValHostPtr[745]=-5.45;
    cooRowIndexHostPtr[746]=187; cooColIndexHostPtr[746]=186; cooValHostPtr[746]=155.0365680897376;
    cooRowIndexHostPtr[747]=187; cooColIndexHostPtr[747]=187; cooValHostPtr[747]=-314.0731361794753;
    cooRowIndexHostPtr[748]=187; cooColIndexHostPtr[748]=188; cooValHostPtr[748]=155.0365680897376;
    cooRowIndexHostPtr[749]=188; cooColIndexHostPtr[749]=88; cooValHostPtr[749]=-5.45;
    cooRowIndexHostPtr[750]=188; cooColIndexHostPtr[750]=187; cooValHostPtr[750]=155.0365680897376;
    cooRowIndexHostPtr[751]=188; cooColIndexHostPtr[751]=188; cooValHostPtr[751]=-314.0731361794753;
    cooRowIndexHostPtr[752]=188; cooColIndexHostPtr[752]=189; cooValHostPtr[752]=155.0365680897376;
    cooRowIndexHostPtr[753]=189; cooColIndexHostPtr[753]=89; cooValHostPtr[753]=-5.45;
    cooRowIndexHostPtr[754]=189; cooColIndexHostPtr[754]=188; cooValHostPtr[754]=155.0365680897376;
    cooRowIndexHostPtr[755]=189; cooColIndexHostPtr[755]=189; cooValHostPtr[755]=-314.0731361794753;
    cooRowIndexHostPtr[756]=189; cooColIndexHostPtr[756]=190; cooValHostPtr[756]=155.0365680897376;
    cooRowIndexHostPtr[757]=190; cooColIndexHostPtr[757]=90; cooValHostPtr[757]=-5.45;
    cooRowIndexHostPtr[758]=190; cooColIndexHostPtr[758]=189; cooValHostPtr[758]=155.0365680897376;
    cooRowIndexHostPtr[759]=190; cooColIndexHostPtr[759]=190; cooValHostPtr[759]=-314.0731361794753;
    cooRowIndexHostPtr[760]=190; cooColIndexHostPtr[760]=191; cooValHostPtr[760]=155.0365680897376;
    cooRowIndexHostPtr[761]=191; cooColIndexHostPtr[761]=91; cooValHostPtr[761]=-5.45;
    cooRowIndexHostPtr[762]=191; cooColIndexHostPtr[762]=190; cooValHostPtr[762]=155.0365680897376;
    cooRowIndexHostPtr[763]=191; cooColIndexHostPtr[763]=191; cooValHostPtr[763]=-314.0731361794753;
    cooRowIndexHostPtr[764]=191; cooColIndexHostPtr[764]=192; cooValHostPtr[764]=155.0365680897376;
    cooRowIndexHostPtr[765]=192; cooColIndexHostPtr[765]=92; cooValHostPtr[765]=-5.45;
    cooRowIndexHostPtr[766]=192; cooColIndexHostPtr[766]=191; cooValHostPtr[766]=155.0365680897376;
    cooRowIndexHostPtr[767]=192; cooColIndexHostPtr[767]=192; cooValHostPtr[767]=-314.0731361794753;
    cooRowIndexHostPtr[768]=192; cooColIndexHostPtr[768]=193; cooValHostPtr[768]=155.0365680897376;
    cooRowIndexHostPtr[769]=193; cooColIndexHostPtr[769]=93; cooValHostPtr[769]=-5.45;
    cooRowIndexHostPtr[770]=193; cooColIndexHostPtr[770]=192; cooValHostPtr[770]=155.0365680897376;
    cooRowIndexHostPtr[771]=193; cooColIndexHostPtr[771]=193; cooValHostPtr[771]=-314.0731361794753;
    cooRowIndexHostPtr[772]=193; cooColIndexHostPtr[772]=194; cooValHostPtr[772]=155.0365680897376;
    cooRowIndexHostPtr[773]=194; cooColIndexHostPtr[773]=94; cooValHostPtr[773]=-5.45;
    cooRowIndexHostPtr[774]=194; cooColIndexHostPtr[774]=193; cooValHostPtr[774]=155.0365680897376;
    cooRowIndexHostPtr[775]=194; cooColIndexHostPtr[775]=194; cooValHostPtr[775]=-314.0731361794753;
    cooRowIndexHostPtr[776]=194; cooColIndexHostPtr[776]=195; cooValHostPtr[776]=155.0365680897376;
    cooRowIndexHostPtr[777]=195; cooColIndexHostPtr[777]=95; cooValHostPtr[777]=-5.45;
    cooRowIndexHostPtr[778]=195; cooColIndexHostPtr[778]=194; cooValHostPtr[778]=155.0365680897376;
    cooRowIndexHostPtr[779]=195; cooColIndexHostPtr[779]=195; cooValHostPtr[779]=-314.0731361794753;
    cooRowIndexHostPtr[780]=195; cooColIndexHostPtr[780]=196; cooValHostPtr[780]=155.0365680897376;
    cooRowIndexHostPtr[781]=196; cooColIndexHostPtr[781]=96; cooValHostPtr[781]=-5.45;
    cooRowIndexHostPtr[782]=196; cooColIndexHostPtr[782]=195; cooValHostPtr[782]=155.0365680897376;
    cooRowIndexHostPtr[783]=196; cooColIndexHostPtr[783]=196; cooValHostPtr[783]=-314.0731361794753;
    cooRowIndexHostPtr[784]=196; cooColIndexHostPtr[784]=197; cooValHostPtr[784]=155.0365680897376;
    cooRowIndexHostPtr[785]=197; cooColIndexHostPtr[785]=97; cooValHostPtr[785]=-5.45;
    cooRowIndexHostPtr[786]=197; cooColIndexHostPtr[786]=196; cooValHostPtr[786]=155.0365680897376;
    cooRowIndexHostPtr[787]=197; cooColIndexHostPtr[787]=197; cooValHostPtr[787]=-314.0731361794753;
    cooRowIndexHostPtr[788]=197; cooColIndexHostPtr[788]=198; cooValHostPtr[788]=155.0365680897376;
    cooRowIndexHostPtr[789]=198; cooColIndexHostPtr[789]=98; cooValHostPtr[789]=-5.45;
    cooRowIndexHostPtr[790]=198; cooColIndexHostPtr[790]=197; cooValHostPtr[790]=155.0365680897376;
    cooRowIndexHostPtr[791]=198; cooColIndexHostPtr[791]=198; cooValHostPtr[791]=-314.0731361794753;
    cooRowIndexHostPtr[792]=198; cooColIndexHostPtr[792]=199; cooValHostPtr[792]=155.0365680897376;
    cooRowIndexHostPtr[793]=199; cooColIndexHostPtr[793]=99; cooValHostPtr[793]=-5.45;
    cooRowIndexHostPtr[794]=199; cooColIndexHostPtr[794]=198; cooValHostPtr[794]=155.0365680897376;
    cooRowIndexHostPtr[795]=199; cooColIndexHostPtr[795]=199; cooValHostPtr[795]=-314.0731361794753;
    
    /*
    //print the matrix
    printf("Input data:\n");
    for (int i=0; i<nnz; i++){
        printf("cooRowIndexHostPtr[%d]=%d  ",i,cooRowIndexHostPtr[i]);
        printf("cooColIndexHostPtr[%d]=%d  ",i,cooColIndexHostPtr[i]);
        printf("cooValHostPtr[%d]=%f     \n",i,cooValHostPtr[i]);
    }
    */

    /* create a dense vector */
    /*  y  = [1.0 2.0 3.0 4.0 5.0] (dense) */
    yHostPtr    = (double *)malloc(n       *sizeof(yHostPtr[0]));
    y_static    = (double *)malloc(n       *sizeof(yHostPtr[0]));
    if(!yHostPtr || !y_static){
        CLEANUP("Host malloc failed (vectors)");
        return 1;
    }

    srand (time(NULL));
    for(int i = 0; i < n; i++){
        y_static[i] = rand() / double(RAND_MAX);
    }

    /*
    //print the vectors
    for (int j=0; j<1; j++){
        for (int i=0; i<n; i++){
            printf("yHostPtr[%d,%d]=%f\n",i,j,yHostPtr[i+n*j]);
        }
    }
    */

    /* allocate GPU memory and copy the matrix and vectors into it */
    cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
    cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    cudaStat3 = cudaMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0]));
    cudaStat4 = cudaMalloc((void**)&y,          n*sizeof(y[0]));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess)) {
        CLEANUP("Device malloc failed");
        return 1;
    }
    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
                           (size_t)(nnz*sizeof(cooRowIndex[0])),
                           cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr,
                           (size_t)(nnz*sizeof(cooColIndex[0])),
                           cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,
                           (size_t)(nnz*sizeof(cooVal[0])),
                           cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y,           y_static,
                           (size_t)(n*sizeof(y[0])),
                           cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess)) {
        CLEANUP("Memcpy from Host to Device failed");
        return 1;
    }

    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
    cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return 1;
    }
    status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
                             csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
        return 1;
    }
    //csrRowPtr = [0 3 4 7 9]

    int devId;
    cudaDeviceProp prop;
    cudaError_t cudaStat;
    cudaStat = cudaGetDevice(&devId);
    if (cudaSuccess != cudaStat){
        CLEANUP("cudaGetDevice failed");
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }
    cudaStat = cudaGetDeviceProperties( &prop, devId) ;
    if (cudaSuccess != cudaStat){
        CLEANUP("cudaGetDeviceProperties failed");
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }

    /* SpmV */
    std::ofstream myfile;
    myfile.open ("example.txt");   
    printf("SpMV elapsed time:\n");
    for(int i = 0; i < 1000; i++){
        srand (time(NULL));
        for(int i = 0; i < n; i++){
            y_static[i] = rand() / double(RAND_MAX);
        }

        cudaMemcpy(y, y_static, (size_t)(n*sizeof(y[0])), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                            &done, descr, cooVal, csrRowPtr, cooColIndex,
                            y, &dzero, y);
        cudaEventRecord(stop);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP("Matrix-vector multiplication failed");
            return 1;
        }
        cudaMemcpy(yHostPtr, y, (size_t)(n*sizeof(y[0])), cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float milliseconds = -1;
        cudaEventElapsedTime(&milliseconds, start, stop); 
        myfile << 1000.0 * milliseconds << "\n";
        cudaDeviceSynchronize();
    }
    myfile.close();

    /* destroy matrix descriptor */
    status = cusparseDestroyMatDescr(descr);
    descr = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor destruction failed");
        return 1;
    }

    /* destroy handle */
    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library release of resources failed");
        return 1;
    }

    printf("SpMV results:\n");
    for (int j=0; j<1; j++){
        for (int i=0; i<n; i++){
            printf("yHostPtr[%d,%d]=%f\n",i,j,yHostPtr[i+n*j]);
        }
    }
    
    CLEANUP("example test PASSED");
    return 0;
}

