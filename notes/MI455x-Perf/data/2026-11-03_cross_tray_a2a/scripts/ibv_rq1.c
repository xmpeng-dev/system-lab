// NCCL_IBVERBS_LIB shim: the frontend ionic NIC rejects ibv_create_qp with max_recv_wr == 0 (EINVAL), which RCCL
// uses for its sender and flush QPs. Raise it to 1; every other verb resolves from libibverbs (DT_NEEDED).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <stdio.h>

typedef struct ibv_qp* (*create_qp_t)(struct ibv_pd*, struct ibv_qp_init_attr*);

struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* attr) {
    static create_qp_t real;
    static int bumped;
    if (!real) {
        void* h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
        real = (create_qp_t)dlvsym(h, "ibv_create_qp", "IBVERBS_1.1");
    }
    if (attr && !attr->srq && attr->cap.max_recv_wr == 0) {
        attr->cap.max_recv_wr = 1;
        if (!bumped++) fprintf(stderr, "ibv_rq1: max_recv_wr 0 -> 1\n");
    }
    return real(pd, attr);
}
