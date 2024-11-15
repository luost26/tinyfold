#ifndef FOLDING_GEOMETRIC_H
#define FOLDING_GEOMETRIC_H

#include <cmath>


inline void apply_affine(const float * affine_4x4, const float * vec3, float * out3) {
    float x = affine_4x4[0] * vec3[0] + affine_4x4[1] * vec3[1] + affine_4x4[2] * vec3[2] + affine_4x4[3];
    float y = affine_4x4[4] * vec3[0] + affine_4x4[5] * vec3[1] + affine_4x4[6] * vec3[2] + affine_4x4[7];
    float z = affine_4x4[8] * vec3[0] + affine_4x4[9] * vec3[1] + affine_4x4[10] * vec3[2] + affine_4x4[11];
    out3[0] = x;
    out3[1] = y;
    out3[2] = z;
}


inline void invert_apply_affine(const float * affine_4x4, const float * vec3, float * out3) {
    float x = vec3[0] - affine_4x4[3];
    float y = vec3[1] - affine_4x4[7];
    float z = vec3[2] - affine_4x4[11];
    out3[0] = affine_4x4[0] * x + affine_4x4[4] * y + affine_4x4[8] * z;
    out3[1] = affine_4x4[1] * x + affine_4x4[5] * y + affine_4x4[9] * z;
    out3[2] = affine_4x4[2] * x + affine_4x4[6] * y + affine_4x4[10] * z;
}


inline void normalize_quat_(float *quat) {
    float length = 0.0f;
    for (int i = 0; i < 4; i ++) {
        length += quat[i] * quat[i];
    }
    length = sqrtf(length);
    for (int i = 0; i < 4; i ++) {
        quat[i] /= length;
    }   
}


inline void standarize_quat_(float *quat) {
    if (quat[0] < 0) {
        for (int i = 0; i < 4; i ++) {
            quat[i] = -quat[i];
        }
    }
}


inline void apply_affine_rotation_only(const float *r, const float *v, float *v_new) {
    float x = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    float y = r[4] * v[0] + r[5] * v[1] + r[6] * v[2];
    float z = r[8] * v[0] + r[9] * v[1] + r[10] * v[2];
    v_new[0] = x;
    v_new[1] = y;
    v_new[2] = z;
}


inline void compose_rotation(const float *affine1, const float *affine2, float *out) {
    float new_rot[9];
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            new_rot[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k ++) {
                new_rot[i * 3 + j] += affine1[i * 4 + k] * affine2[k * 4 + j];
            }
        }
    }
    out[0] = new_rot[0]; out[1] = new_rot[1]; out[2] = new_rot[2];
    out[4] = new_rot[3]; out[5] = new_rot[4]; out[6] = new_rot[5];
    out[8] = new_rot[6]; out[9] = new_rot[7]; out[10] = new_rot[8];
}


inline void compose_affine(const float *affine1, const float *affine2, float *out) {
    float new_affine[16];
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) {
            new_affine[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; k ++) {
                new_affine[i * 4 + j] += affine1[i * 4 + k] * affine2[k * 4 + j];
            }
        }
    }
    for (int i = 0; i < 16; i ++) {
        out[i] = new_affine[i];
    }
}

inline void affine_to_quat(const float *r, float *quat) {
    float m00 = r[0], m01 = r[1], m02 = r[2];
    float m10 = r[4], m11 = r[5], m12 = r[6];
    float m20 = r[8], m21 = r[9], m22 = r[10];
    #define SQRT_RELU(x) (sqrtf(fmaxf(0.0f, x)))
    float q_abs[4] = {
        SQRT_RELU(1.0 + m00 + m11 + m22),
        SQRT_RELU(1.0 + m00 - m11 - m22),
        SQRT_RELU(1.0 - m00 + m11 - m22),
        SQRT_RELU(1.0 - m00 - m11 + m22)
    };

    int best_q_idx = 0;
    for (int i = 1; i < 4; i ++) {
        if (q_abs[i] > q_abs[best_q_idx]) {
            best_q_idx = i;
        }
    }

    #define UNPACK_QUAT(out,q1,q2,q3,q4) { out[0] = q1; out[1] = q2; out[2] = q3; out[3] = q4; }
    switch (best_q_idx) {
    case 0:
        UNPACK_QUAT(quat, q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01);
        break;
    case 1:
        UNPACK_QUAT(quat, m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20);
        break;
    case 2:
        UNPACK_QUAT(quat, m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21);
    default:
        UNPACK_QUAT(quat, m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]);
        break;
    }
    #undef UNPACK_QUAT

    normalize_quat_(quat);
    standarize_quat_(quat);
}


inline void quat_to_affine(const float *quat, float *affine) {
    float r = quat[0], i = quat[1], j = quat[2], k = quat[3];
    float two_s = 2.0 / (r * r + i * i + j * j + k * k);
    // Row 0
    affine[0] = 1 - two_s * (j * j + k * k);
    affine[1] = two_s * (i * j - k * r);
    affine[2] = two_s * (i * k + j * r);
    // Row 1
    affine[4] = two_s * (i * j + k * r);
    affine[5] = 1 - two_s * (i * i + k * k);
    affine[6] = two_s * (j * k - i * r);
    // Row 2
    affine[8] = two_s * (i * k - j * r);
    affine[9] = two_s * (j * k + i * r);
    affine[10] = 1 - two_s * (i * i + j * j);
}


inline float dihedral_from_four_points(const float *p0, const float *p1, const float *p2, const float *p3) {
    #define INIT_VEC_MINUS(A,B) {A[0] - B[0], A[1] - B[1], A[2] - B[2]}
    #define INIT_VEC_CROSS(A,B) {A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]}
    #define NORMALIZE_(VEC) { float norm = sqrtf(VEC[0] * VEC[0] + VEC[1] * VEC[1] + VEC[2] * VEC[2]); VEC[0] /= norm; VEC[1] /= norm; VEC[2] /= norm; }
    #define INNERPROD(A,B) (A[0] * B[0] + A[1] * B[1] + A[2] * B[2])

    float v0[3] = INIT_VEC_MINUS(p2, p1);
    float v1[3] = INIT_VEC_MINUS(p0, p1);
    float v2[3] = INIT_VEC_MINUS(p3, p2);
    
    float n1[3] = INIT_VEC_CROSS(v0, v1);
    NORMALIZE_(n1);

    float n2[3] = INIT_VEC_CROSS(v0, v2);
    NORMALIZE_(n2);

    float u3[3] = INIT_VEC_CROSS(v1, v2);
    float sgn = INNERPROD(u3, v0) >= 0 ? 1.0f : -1.0f;

    float cos_theta = INNERPROD(n1, n2);
    if (cos_theta > 0.999999) {
        cos_theta = 0.999999;
    } else if (cos_theta < -0.999999) {
        cos_theta = -0.999999;
    }
    float dihed = sgn * acosf(cos_theta);
    if (dihed != dihed) {
        dihed = 0.0f;
    }

    #undef INIT_VEC_MINUS
    #undef INIT_VEC_CROSS
    #undef NORMALIZE_
    #undef INNERPROD

    return dihed;
}

#endif