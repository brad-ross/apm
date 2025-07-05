#include <gtest/gtest.h>
#include "apm.h"

TEST(APMTest, GetVersionTest) {
    std::string version = apm::get_version();
    EXPECT_EQ(version, "0.1.0");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 