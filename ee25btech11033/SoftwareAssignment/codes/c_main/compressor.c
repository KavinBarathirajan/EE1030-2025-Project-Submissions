#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

/*------------ STB LIBRARY IMPLEMENTATION ------------*/
#define STB_IMAGE_IMPLEMENTATION
#include "../c_libs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../c_libs/stb_image_write.h"


/*------------ MATRIX FUNCTIONS ------------*/
double **matrix_transpose(double  **A, int m, int n) {
    double **AT;
    AT = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        AT[i] = (double *)malloc(m * sizeof(double));
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

double **matrix_multiply(double **A, double **B, int m, int n, int p){
    double **C;
    C = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        C[i] = (double *)malloc(p * sizeof(double));
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

double **identity_matrix(int n) {
    double **I;
    I = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        I[i] = (double *)malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) { 
            if (i == j) {
                I[i][j] = 1.0;
            } 
            else {
                I[i][j] = 0.0;
            }
        }
    }
    return I;
}

double fnorm(double **A, int m, int n){
    double sum = 0.0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            sum += A[i][j] * A[i][j];   
        }
    }
    return sqrt(sum);
}


/*------------ JACOBI EIGENVALUE ALGORITHM ------------*/
void eigen_system(double **A, int n, double *eigenvals, double **V){
    double **B;
    B = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        B[i] = (double *)malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = A[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double tol = 1e-10;
    int max_iters = 100*n;
for (int iter = 0; iter < max_iters; iter++) {
        // Find largest off-diagonal element (p, q)
        int p = 0, q = 1;
        double max_val = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(B[i][j]) > max_val) {
                    max_val = fabs(B[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if (max_val < tol) {
            break;
        }

        // Compute rotation parameters
        double B_pp = B[p][p];
        double B_qq = B[q][q];
        double B_pq = B[p][q];

        double tau = (B_qq - B_pp) / (2.0 * B_pq);
        double t;
        if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1.0 + tau * tau));
        } 
        else {
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
        }
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        // Optimized B update (only affects rows/cols p and q)
        B[p][p] = c * c * B_pp + s * s * B_qq - 2.0 * c * s * B_pq;
        B[q][q] = s * s * B_pp + c * c * B_qq + 2.0 * c * s * B_pq;
        B[p][q] = 0.0;
        B[q][p] = 0.0;

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double B_ip = B[i][p];
                double B_iq = B[i][q];
                B[i][p] = c * B_ip - s * B_iq;
                B[i][q] = s * B_ip + c * B_iq;
                B[p][i] = B[i][p]; 
                B[q][i] = B[i][q]; 
            }
        }

        // Optimized Eigenvector update (only affects cols p and q)
        for (int i = 0; i < n; i++) {
            double V_ip = V[i][p];
            double V_iq = V[i][q];
            V[i][p] = c * V_ip - s * V_iq;
            V[i][q] = s * V_ip + c * V_iq;
        }
    }

    // Extract eigenvalues from diagonal matrix
    for (int i = 0; i < n; i++) {
        eigenvals[i] = B[i][i];
    }

    for (int i = 0; i < n; i++) {
        free(B[i]);
    }
    free(B);
}

void sort(double *B, int n, double **V){
    for(int i = 0; i < n - 1; i++){
        for(int j = 0; j < n - i - 1; j++){
            if(B[j] < B[j + 1]){
                double temp = B[j];
                B[j] = B[j + 1];
                B[j + 1] = temp;
                //Also swap eigen vectors accordingly
                for(int k = 0; k < n; k++){
                    double temp_v = V[k][j];
                    V[k][j] = V[k][j + 1];
                    V[k][j + 1] = temp_v;
                }
            }
        }
    }
}


/*------------ TRUNCATED SVD FUNCTION ------------*/
double **truncated_SVD(double **A, int m, int n, int k){
    double **AT = matrix_transpose(A, m, n);
    double **ATA = matrix_multiply(AT, A, n, m, n);

    double *eigenvals = (double *)malloc(n * sizeof(double));
    double *singularvals = (double *)malloc(n * sizeof(double));
    double **V = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++){ 
        V[i] = (double *)malloc(n * sizeof(double));
    }


    eigen_system(ATA, n, eigenvals, V);

    // Convert eigenvalues to singular values
    for(int i = 0; i < n; i++){
        if(eigenvals[i] < 0){
            singularvals[i] = 0.0;
        }
        else{
            singularvals[i] = sqrt(eigenvals[i]);
        }
    }

    sort(singularvals, n, V);
    
    //Construct sigma_trunc (dim k x k)
    double **sigma_trunc = identity_matrix(k);
    for(int i = 0; i < k; i++){
        sigma_trunc[i][i] = singularvals[i];
    }

    //Construct VT_trunc (dim k x n)
    double **V_trunc;
    V_trunc = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        V_trunc[i] = (double *)malloc(k * sizeof(double));
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++){
            V_trunc[i][j] = V[i][j];
        }
    }
    double **VT_trunc = matrix_transpose(V_trunc, n, k);

    //Compute U_trunc (dim m x k)
    double **U_trunc;
    U_trunc = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        U_trunc[i] = (double *)malloc(k * sizeof(double));
    }
    for (int j = 0; j < k; j++) {
        double sigma = singularvals[j];
        if(fabs(sigma) > 1e-10){
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * V[l][j];
                }
                U_trunc[i][j] = sum / sigma;
            }
        }
        else{
            for (int i = 0; i < m; i++) {
                U_trunc[i][j] = 0.0;
            }
        }
    }
    
    //Compute A_trunc (dim m x n)
    double **A_trunc_temp = matrix_multiply(U_trunc, sigma_trunc, m, k, k);
    double **A_trunc = matrix_multiply(A_trunc_temp, VT_trunc, m, k, n);
    
    for(int i = 0; i < m; i++){
        free(U_trunc[i]);
    }
    free(U_trunc);
    
    for(int i = 0; i < n; i++){
        free(V[i]);
    }
    free(V);
    for(int i = 0; i < n; i++){
        free(V_trunc[i]);
    }   
    free(V_trunc);
    for(int i = 0; i < k; i++){
        free(sigma_trunc[i]);
    }
    free(sigma_trunc);
    for(int i = 0; i < m; i++){
        free(A_trunc_temp[i]);
    }
    free(A_trunc_temp);
    free(singularvals);
    for(int i = 0; i < n; i++){
        free(AT[i]);
    }
    free(AT);
    for(int i = 0; i < n; i++){
        free(ATA[i]);
    }
    free(ATA);
    for(int i = 0; i < k; i++){
        free(VT_trunc[i]);
    }
    free(VT_trunc);
    free(eigenvals);
    return A_trunc;
}



/*------------ IMAGE IMPLEMENTATION FUNCTIONS ------------*/
unsigned char **img_to_matrix(const char *filename, int *width, int *height) {
    int original_channels;
    int desired_channels = 1;

    unsigned char *data_1d = stbi_load(filename, width, height, &original_channels, desired_channels);

    if (data_1d == NULL) {
        return NULL;
    }

    int w = *width;
    int h = *height;

    unsigned char **img = (unsigned char **)malloc(h * sizeof(unsigned char *));
    if (!img) {
         stbi_image_free(data_1d);
         return NULL;
    }

    for (int i = 0; i < h; i++) {
        img[i] = (unsigned char *)malloc(w * sizeof(unsigned char));
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            img[i][j] = data_1d[i * w + j];
        }
    }
    stbi_image_free(data_1d);
    return img;
}

void matrix_to_jpg(const char *filename, unsigned char **A, int width, int height) {
    int channels = 1;
    int quality = 45;
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (data == NULL) {
         fprintf(stderr, "Error: Memory allocation failed while saving JPG.\n");
         return;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = A[i][j];
        }
    }
    stbi_write_jpg(filename, width, height, channels, data, quality);
    free(data);
}

void matrix_to_png(const char *filename, unsigned char **A, int width, int height) {
    int channels = 1;
    int stride = width * channels;
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (data == NULL) {
         fprintf(stderr, "Error: Memory allocation failed while saving PNG.\n");
         return;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = A[i][j];
        }
    }

    stbi_write_png(filename, width, height, channels, data, stride);
    free(data);
}

const char *extension(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (dot) {
        return dot + 1;
    }
    return "";
}

int isequal(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a++) != tolower((unsigned char)*b++)) return 0;
    }
    return *a == *b;
}


/*------------ MAIN FUNCTION ------------*/

int main() {
    int width, height;

    const char *input_filename = "../../figs/einstein.jpg";
    const char *output_filename = "../../figs/einstein_100.jpg";

    unsigned char **img = img_to_matrix(input_filename, &width, &height);
    if(img == NULL){
        printf("Error loading image.\n");
        return -1;
    }

    //Convert unsigned char matrix to double matrix for SVD
    double **A;
    A = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        A[i] = (double *)malloc(width * sizeof(double));
    }
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            A[i][j] = (double)img[i][j];
        }
    }

    //Set value of truncation k
    int k = 100;

    //Check the value of k
    if (k < 1){
        k = 1;
    }
    int mn = (width < height)? width : height;
    if (k > mn){ 
        k = mn;
    }

    double **A_trunc = truncated_SVD(A, height, width, k);

    //Convert double matrix to unsigned char matrix
    unsigned char **img_trunc = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (int i = 0; i < height; i++) {
        img_trunc[i] = (unsigned char *)malloc(width * sizeof(unsigned char));
    }

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(A_trunc[i][j] < 0.0) A_trunc[i][j] = 0.0;
            if(A_trunc[i][j] > 255.0) A_trunc[i][j] = 255.0;
            img_trunc[i][j] = (unsigned char)(A_trunc[i][j]);
        }
    }

    const char *ext = extension(output_filename);

    if (isequal(ext, "jpg") || isequal(ext, "jpeg")) {
        matrix_to_jpg(output_filename, img_trunc, width, height);
    }
    else if (isequal(ext, "png")) {
        matrix_to_png(output_filename, img_trunc, width, height);
    }
    else {
        printf("Unsupported output format: .%s\n", ext);
    }

    double **A_diff;
    A_diff = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        A_diff[i] = (double *)malloc(width * sizeof(double));
    }
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            A_diff[i][j] = A[i][j] - A_trunc[i][j];
        }
    }
    printf("Image compressed using k = %d\n", k);
    printf("Frobenius norm : %lf\n", fnorm(A_diff, height, width));
    
    double error = fnorm(A_diff, height, width) / fnorm(A, height, width) * 100.0;
    printf("Percentage error (Frobenius norm): %lf\n", error);

    for(int i = 0; i < height; i++){
        free(img[i]);
        free(A[i]); 
        free(A_trunc[i]);
        free(img_trunc[i]);
    }
    free(img);
    free(A);
    free(A_trunc);
    free(img_trunc);
    for(int i = 0; i < height; i++){
        free(A_diff[i]);
    }
    free(A_diff);
    return 0;
}
