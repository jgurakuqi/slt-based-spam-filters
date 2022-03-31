#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <tuple>
#include <atomic>
#include <condition_variable>
#include <exception>

struct thread_provider_exception : public std::exception
{
    const char *what() const throw()
    {
        return "thread_provider_exception: Cannot start a thread pool with less then 1 thread!";
    }
};

/**
 * @brief The following class defines a thread library whose aim is to create a persistent pool of threads,
 * allowing for the assignement of new tasks without the need to manually create and join each single thread.
 *
 */
template <class memory_protection_method>
class thread_provider
{
private:
    std::vector<std::thread> threads_pool;
    std::queue<std::function<void()>> tasks_queue;
    std::mutex pool_mutex;
    std::mutex queue_mutex;
    std::condition_variable pool_condition;
    bool pool_termination;

    /**
     * @brief The following method creates as many threads as specified by the threads_number param.
     *
     * @param threads_number
     */
    void createThreads(int threads_number)
    {
        pool_termination = false;
        if (threads_number > 0)
        {
            for (int i = 0; i < threads_number; i++)
            {
                threads_pool.push_back(
                    std::thread(&thread_provider::waitForTasks, this));
            }
        }
        else
        {
            throw thread_provider_exception();
        }
    }

    /**
     * @brief The following method keeps each created thread on idle, waiting for a new task to be executed.
     * When a new task becomes available, one of the idle threads will be awakened in order to process the given task.
     *
     */
    void waitForTasks()
    {
        std::function<void()> taskToPerform;
        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                pool_condition.wait(lock, [&]
                                    { return !tasks_queue.empty() || pool_termination; });
                if (!tasks_queue.empty())
                {
                    taskToPerform = tasks_queue.front();
                    tasks_queue.pop();
                }
                else
                {
                    return;
                }
            }
            if (taskToPerform != nullptr)
            {
                taskToPerform();
            }
        }
    }

public:
    memory_protection_method protection;

    thread_provider(const thread_provider &) = delete;
    thread_provider &operator=(const thread_provider &) = delete;

    /**
     * @brief The following constructor launches the creation of as many threads
     * as the hardware concurrency of the running machine.
     *
     */
    thread_provider()
    {
        createThreads(std::thread::hardware_concurrency());
    }

    /**
     * @brief The following constructor launches the creation of as many threads
     * as specified by the argument.
     *
     * @param threads_number
     */
    thread_provider(int threads_number)
    {
        createThreads(threads_number);
    }

    /**
     * @brief The following method returns the number of created threads.
     *
     * @return int
     */
    int getPoolSize()
    {
        return threads_pool.size();
    }

    /**
     * @brief The following method pushes the new task into the queue of tasks, so to execute it as soon as a thread
     * will be on idle.
     *
     * @param threadIndex
     */
    void executeTask(std::function<void()> &&newTask)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks_queue.push(std::function<void()>(newTask));
        }
        pool_condition.notify_one();
    }

    /**
     * @brief The following method shutdowns the pool of threads.
     *
     */
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> thMutex(pool_mutex);
            pool_termination = true;
        }
        pool_condition.notify_all();

        // Join all threads.
        for (std::thread &actThread : threads_pool)
        {
            actThread.join();
        }
    }
};