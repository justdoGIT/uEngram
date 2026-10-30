/* LD_PRELOAD shim that logs the file operations a process performs under one directory.
 *
 * Build: gcc -shared -fPIC -O2 -o libcrashshim.so bench/crash_shim.c -ldl
 * Use:   UENGRAM_CC_ROOT=<dir> UENGRAM_CC_LOG=<file> LD_PRELOAD=./libcrashshim.so python ...
 *
 * Every completed write, fsync/fdatasync, directory fsync, truncate, create and unlink on a path
 * under UENGRAM_CC_ROOT is appended to the log in call order. bench/capsule_crash_model.py replays the
 * log to rebuild the disk image a POSIX-adversarial filesystem could leave after a power cut at any
 * point. The shim only observes: every call is passed through unchanged.
 *
 * Record: u8 op | u32 pathlen | path | u64 a | u64 b | u32 datalen | data   (little endian)
 *   1 WRITE a=offset            2 SYNC            3 TRUNC a=length (also fallocate extension)
 *   4 CREATE                    5 UNLINK          6 DIRSYNC        7 MARK (data = text, path empty)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

enum { OP_WRITE = 1, OP_SYNC, OP_TRUNC, OP_CREATE, OP_UNLINK, OP_DIRSYNC, OP_MARK };

static int log_fd = -1;
static char root[PATH_MAX];
static size_t root_len;
static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;

static void *resolve(void **cache, const char *name) {
    if (!*cache) *cache = dlsym(RTLD_NEXT, name);
    return *cache;
}
#define REAL(name) ((typeof(real_##name))resolve((void **)&real_##name, #name))
static int (*real_open)(const char *, int, ...);
static int (*real_open64)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_openat64)(int, const char *, int, ...);
static ssize_t (*real_write)(int, const void *, size_t);
static ssize_t (*real_pwrite)(int, const void *, size_t, off_t);
static ssize_t (*real_pwrite64)(int, const void *, size_t, off64_t);
static int (*real_fsync)(int);
static int (*real_fdatasync)(int);
static int (*real_ftruncate)(int, off_t);
static int (*real_ftruncate64)(int, off64_t);
static int (*real_posix_fallocate)(int, off_t, off_t);
static int (*real_posix_fallocate64)(int, off64_t, off64_t);
static int (*real_unlink)(const char *);
static off_t (*real_lseek)(int, off_t, int);

__attribute__((constructor)) static void shim_init(void) {
    const char *r = getenv("UENGRAM_CC_ROOT"), *l = getenv("UENGRAM_CC_LOG");
    if (!r || !l) return;
    strncpy(root, r, sizeof root - 1);
    root_len = strlen(root);
    log_fd = (int)syscall(SYS_openat, AT_FDCWD, l, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
}

static int under_root(const char *path) {
    return log_fd >= 0 && path && strncmp(path, root, root_len) == 0 &&
           (path[root_len] == '/' || path[root_len] == '\0');
}

static int fd_path(int fd, char *out, size_t cap) {
    char link[64];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t n = syscall(SYS_readlink, link, out, cap - 1);
    if (n <= 0) return 0;
    out[n] = '\0';
    return under_root(out);
}

static void emit(uint8_t op, const char *path, uint64_t a, uint64_t b, const void *data, uint32_t dlen) {
    uint32_t plen = path ? (uint32_t)strlen(path) : 0;
    size_t total = 1 + 4 + plen + 8 + 8 + 4 + dlen;
    uint8_t *buf = malloc(total);
    if (!buf) return;
    uint8_t *p = buf;
    *p++ = op;
    memcpy(p, &plen, 4), p += 4;
    if (plen) memcpy(p, path, plen), p += plen;
    memcpy(p, &a, 8), p += 8;
    memcpy(p, &b, 8), p += 8;
    memcpy(p, &dlen, 4), p += 4;
    if (dlen) memcpy(p, data, dlen);
    pthread_mutex_lock(&log_lock);
    syscall(SYS_write, log_fd, buf, total);
    pthread_mutex_unlock(&log_lock);
    free(buf);
}

void uengram_cc_mark(const char *text) {
    if (log_fd >= 0) emit(OP_MARK, NULL, 0, 0, text, (uint32_t)strlen(text));
}

static int is_new_file(const char *path, int flags) {
    if (!(flags & O_CREAT) || !path || path[0] != '/' || !under_root(path)) return 0;
    return syscall(SYS_faccessat, AT_FDCWD, path, F_OK, 0) != 0;
}
static int existed_before(const char *path) {
    return path && path[0] == '/' && under_root(path) && syscall(SYS_faccessat, AT_FDCWD, path, F_OK, 0) == 0;
}

#define OPEN_BODY(CALL, PATHARG)                                                              \
    mode_t mode = 0;                                                                          \
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); } \
    int fresh = is_new_file(PATHARG, flags);                                                  \
    int truncating = (flags & O_TRUNC) ? existed_before(PATHARG) : 0;                         \
    int fd = CALL;                                                                            \
    if (fd >= 0 && fresh) emit(OP_CREATE, PATHARG, 0, 0, NULL, 0);                            \
    if (fd >= 0 && truncating) emit(OP_TRUNC, PATHARG, 0, 0, NULL, 0);                        \
    return fd;

int open(const char *path, int flags, ...) { OPEN_BODY(REAL(open)(path, flags, mode), path) }
int open64(const char *path, int flags, ...) { OPEN_BODY(REAL(open64)(path, flags, mode), path) }
int openat(int dirfd, const char *path, int flags, ...) {
    OPEN_BODY(REAL(openat)(dirfd, path, flags, mode), path)
}
int openat64(int dirfd, const char *path, int flags, ...) {
    OPEN_BODY(REAL(openat64)(dirfd, path, flags, mode), path)
}

static void log_write(int fd, const void *buf, ssize_t n, int64_t offset) {
    char path[PATH_MAX];
    if (n > 0 && fd_path(fd, path, sizeof path)) emit(OP_WRITE, path, (uint64_t)offset, 0, buf, (uint32_t)n);
}

ssize_t write(int fd, const void *buf, size_t count) {
    off_t at = REAL(lseek)(fd, 0, SEEK_CUR);
    if (fcntl(fd, F_GETFL) & O_APPEND) {
        struct stat st;
        if (fstat(fd, &st) == 0) at = st.st_size;
    }
    ssize_t n = REAL(write)(fd, buf, count);
    if (n > 0 && at >= 0) log_write(fd, buf, n, at);
    return n;
}

ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset) {
    ssize_t n = REAL(pwrite)(fd, buf, count, offset);
    log_write(fd, buf, n, offset);
    return n;
}

ssize_t pwrite64(int fd, const void *buf, size_t count, off64_t offset) {
    ssize_t n = REAL(pwrite64)(fd, buf, count, offset);
    log_write(fd, buf, n, offset);
    return n;
}

static void log_sync(int fd) {
    char path[PATH_MAX];
    struct stat st;
    if (!fd_path(fd, path, sizeof path) || fstat(fd, &st) != 0) return;
    emit(S_ISDIR(st.st_mode) ? OP_DIRSYNC : OP_SYNC, path, 0, 0, NULL, 0);
}

int fsync(int fd) {
    int rc = REAL(fsync)(fd);
    if (rc == 0) log_sync(fd);
    return rc;
}

int fdatasync(int fd) {
    int rc = REAL(fdatasync)(fd);
    if (rc == 0) log_sync(fd);
    return rc;
}

static void log_trunc(int fd, uint64_t length) {
    char path[PATH_MAX];
    if (fd_path(fd, path, sizeof path)) emit(OP_TRUNC, path, length, 0, NULL, 0);
}

int ftruncate(int fd, off_t length) {
    int rc = REAL(ftruncate)(fd, length);
    if (rc == 0) log_trunc(fd, (uint64_t)length);
    return rc;
}

int ftruncate64(int fd, off64_t length) {
    int rc = REAL(ftruncate64)(fd, length);
    if (rc == 0) log_trunc(fd, (uint64_t)length);
    return rc;
}

int posix_fallocate(int fd, off_t offset, off_t len) {
    struct stat st;
    uint64_t before = fstat(fd, &st) == 0 ? (uint64_t)st.st_size : 0;
    int rc = REAL(posix_fallocate)(fd, offset, len);
    if (rc == 0 && (uint64_t)(offset + len) > before) log_trunc(fd, (uint64_t)(offset + len));
    return rc;
}

int posix_fallocate64(int fd, off64_t offset, off64_t len) {
    struct stat st;
    uint64_t before = fstat(fd, &st) == 0 ? (uint64_t)st.st_size : 0;
    int rc = REAL(posix_fallocate64)(fd, offset, len);
    if (rc == 0 && (uint64_t)(offset + len) > before) log_trunc(fd, (uint64_t)(offset + len));
    return rc;
}

int unlink(const char *path) {
    int tracked = path[0] == '/' && under_root(path);
    int rc = REAL(unlink)(path);
    if (rc == 0 && tracked) emit(OP_UNLINK, path, 0, 0, NULL, 0);
    return rc;
}
