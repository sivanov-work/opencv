// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP

#include <atomic>
#include <memory>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {

class GAPI_EXPORTS SharedLock {
public:
    SharedLock() = default;
    ~SharedLock() = default;

    size_t shared_lock();
    size_t unlock_shared();

    void lock();
    bool try_lock();
    void unlock();

    bool owns() const;
private:
    enum {
        EXCLUSIVE_ACCESS = -1
    };

    SharedLock(const SharedLock&) = delete;
    SharedLock& operator= (const SharedLock&) = delete;
    SharedLock(SharedLock&&) = delete;
    SharedLock& operator== (SharedLock&&) = delete;

    std::atomic<int> counter;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP
