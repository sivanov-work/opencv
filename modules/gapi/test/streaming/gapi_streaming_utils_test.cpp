// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"

#include <chrono>
#include <future>

#define private public
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#undef private

namespace opencv_test
{
namespace
{
using cv::gapi::wip::SharedLock;

TEST(SharedLock, Create) {
    SharedLock lock;
    EXPECT_EQ(lock.counter.load(), 0);
}

TEST(SharedLock, Read_SingleThread)
{
    SharedLock lock;

    const size_t single_thread_read_count = 100;
    for(size_t i = 0; i < single_thread_read_count; i++) {
        lock.shared_lock();
        EXPECT_FALSE(lock.owns());
    }
    EXPECT_EQ(lock.counter.load(), single_thread_read_count);

    for(size_t i = 0; i < single_thread_read_count; i++) {
        lock.unlock_shared();
        EXPECT_FALSE(lock.owns());
    }

    EXPECT_EQ(lock.counter.load(), 0);
}

TEST(SharedLock, TryLock_SingleThread)
{
    SharedLock lock;

    EXPECT_TRUE(lock.try_lock());
    EXPECT_TRUE(lock.owns());

    lock.unlock();
    EXPECT_FALSE(lock.owns());
    EXPECT_EQ(lock.counter.load(), 0);
}

TEST(SharedLock, Write_SingleThread)
{
    SharedLock lock;

    lock.lock();
    EXPECT_TRUE(lock.owns());

    lock.unlock();
    EXPECT_FALSE(lock.owns());
    EXPECT_EQ(lock.counter.load(), 0);
}

TEST(SharedLock, TryLockTryLock_SingleThread)
{
    SharedLock lock;

    lock.try_lock();
    EXPECT_FALSE(lock.try_lock());
    lock.unlock();

    EXPECT_FALSE(lock.owns());
}

TEST(SharedLock, ReadTryLock_SingleThread)
{
    SharedLock lock;

    lock.shared_lock();
    EXPECT_FALSE(lock.owns());
    EXPECT_FALSE(lock.try_lock());
    lock.unlock_shared();

    EXPECT_TRUE(lock.try_lock());
    EXPECT_TRUE(lock.owns());
    lock.unlock();
}

TEST(SharedLock, WriteTryLock_SingleThread)
{
    SharedLock lock;

    lock.lock();
    EXPECT_TRUE(lock.owns());
    EXPECT_FALSE(lock.try_lock());
    lock.unlock();

    EXPECT_TRUE(lock.try_lock());
    EXPECT_TRUE(lock.owns());
    lock.unlock();
}


TEST(SharedLock, Write_MultiThread)
{
    SharedLock lock;

    std::promise<void> barrier;
    std::shared_future<void> sync = barrier.get_future();

    const size_t work_count = 3;
    const size_t inc_count = 10000000;
    size_t shared_value = 0;
    auto work = [&lock, &shared_value](size_t count) {
        for (size_t i = 0; i < count; i ++) {
            lock.lock();
            shared_value ++;
            lock.unlock();
        }
    };

    std::thread worker_thread([&barrier, sync, work, inc_count] () {

        std::thread sub_worker([&barrier, work, inc_count] () {
            barrier.set_value();
            work(inc_count);
        });

        sync.wait();
        work(inc_count);
        sub_worker.join();
    });
    sync.wait();

    work(inc_count);
    worker_thread.join();

    EXPECT_EQ(shared_value, inc_count * 3);
}

TEST(SharedLock, ReadWrite_MultiThread)
{
    SharedLock lock;

    std::promise<void> barrier;
    std::future<void> sync = barrier.get_future();

    const size_t inc_count = 10000000;
    size_t shared_value = 0;
    auto write_work = [&lock, &shared_value](size_t count) {
        for (size_t i = 0; i < count; i ++) {
            lock.lock();
            shared_value ++;
            lock.unlock();
        }
    };

    auto read_work = [&lock, &shared_value](size_t count) {

        auto old_shared_value = shared_value;
        for (size_t i = 0; i < count; i ++) {
            lock.shared_lock();
            EXPECT_TRUE(shared_value >= old_shared_value);
            old_shared_value = shared_value;
            lock.unlock_shared();
        }
    };

    std::thread writer_thread([&barrier, write_work, inc_count] () {
        barrier.set_value();
        write_work(inc_count);
    });
    sync.wait();

    read_work(inc_count);
    writer_thread.join();

    EXPECT_EQ(shared_value, inc_count);
}
}
} // opencv_test
