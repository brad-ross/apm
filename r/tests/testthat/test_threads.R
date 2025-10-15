context("Testing multithreading functionality")

test_that("set_apm_threads updates data.table thread count", {
    # data.table loaded in tests/testthat.R

    orig <- getDTthreads()
    on.exit(setDTthreads(orig), add = TRUE)

    # Skip if the machine/CI can't use >1 threads
    if (parallel::detectCores(logical = TRUE) < 2L) {
        skip("Less than 2 logical cores available; skipping multi-thread test.")
    }

    # Set to 2 threads
    set_apm_threads(2L)
    expect_equal(getDTthreads(), 2L)

    # Set back to 1 thread
    set_apm_threads(1L)
    expect_equal(getDTthreads(), 1L)

    # Set to auto (all logical cores)
    set_apm_threads(NULL)
    expect_gte(getDTthreads(), 1L)
    expect_lte(getDTthreads(), parallel::detectCores(logical = TRUE))
})


