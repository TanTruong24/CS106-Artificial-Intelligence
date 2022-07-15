from ortools.algorithms import pywrapknapsack_solver
from time import process_time
from csv import writer


def read_test(test_path):
    with open(test_path, 'r') as f:
        testcase = f.read()

    lines = testcase.split('\n')

    capacities = [int(lines[2])]
    values = []
    weights = []
    temp_weights = []
    for i in range(4, len(lines)-1):
        lst = lines[i].split()
        values.append(int(lst[0]))
        temp_weights.append(int(lst[1]))

    weights.append(temp_weights)

    return values, weights, capacities


def insert_res_csv(test_path, str_opt, time_s, total_value, total_weight, packed_items, packed_weights):
    record = [test_path, str_opt, time_s, total_value,
              total_weight, packed_items, packed_weights]
    with open('result-130test-7p.csv', 'a', newline='') as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(record)
        # Close the file object
        f_object.close()


def solver(test_path, time_limit):
    values, weights, capacities = read_test(test_path)
    packed_items = []
    packed_weights = []
    total_weight = 0

    start_s = process_time()
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    solver.Init(values, weights, capacities)
    solver.set_time_limit(time_limit)
    computed_value = solver.Solve()
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    end_s = process_time()
    elapsed_time_s = end_s - start_s

    if (elapsed_time_s < time_limit):
        str_opt = 'Optimal'
        print("=> Optimal")
    else:
        str_opt = 'Not Optimal'
        print("=> Not optimal")

    print('Time (s): {}'.format(elapsed_time_s))
    print('Total value =', computed_value)
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights)
    insert_res_csv(test_path, str_opt, elapsed_time_s, computed_value,
                   total_weight, packed_items, packed_weights)


def main():
    # Create the solver.
    path_00Uncorrelated = [
        "./00Uncorrelated/n00050/R10000/s002.kp",
        "./00Uncorrelated/n00100/R10000/s003.kp",
        "./00Uncorrelated/n00200/R10000/s005.kp",
        "./00Uncorrelated/n00500/R10000/s008.kp",
        "./00Uncorrelated/n00500/R01000/s008.kp",
        "./00Uncorrelated/n01000/R01000/s013.kp",
        "./00Uncorrelated/n02000/R01000/s021.kp",
        "./00Uncorrelated/n05000/R01000/s034.kp",
        "./00Uncorrelated/n10000/R01000/s055.kp",
        "./00Uncorrelated/n10000/R10000/s055.kp"
    ]
    path_01WeaklyCorrelated = [
        "./01WeaklyCorrelated/n00050/R10000/s002.kp",
        "./01WeaklyCorrelated/n00100/R10000/s003.kp",
        "./01WeaklyCorrelated/n00200/R10000/s005.kp",
        "./01WeaklyCorrelated/n00500/R10000/s008.kp",
        "./01WeaklyCorrelated/n00500/R01000/s008.kp",
        "./01WeaklyCorrelated/n01000/R01000/s013.kp",
        "./01WeaklyCorrelated/n02000/R01000/s021.kp",
        "./01WeaklyCorrelated/n05000/R01000/s034.kp",
        "./01WeaklyCorrelated/n10000/R01000/s055.kp",
        "./01WeaklyCorrelated/n10000/R10000/s055.kp"
    ]
    path_02StronglyCorrelated = [
        "./02StronglyCorrelated/n00050/R10000/s002.kp",
        "./02StronglyCorrelated/n00100/R10000/s003.kp",
        "./02StronglyCorrelated/n00200/R10000/s005.kp",
        "./02StronglyCorrelated/n00500/R10000/s008.kp",
        "./02StronglyCorrelated/n00500/R01000/s008.kp",
        "./02StronglyCorrelated/n01000/R01000/s013.kp",
        "./02StronglyCorrelated/n02000/R01000/s021.kp",
        "./02StronglyCorrelated/n05000/R01000/s034.kp",
        "./02StronglyCorrelated/n10000/R01000/s055.kp",
        "./02StronglyCorrelated/n10000/R10000/s055.kp"
    ]
    path_03InverseStronglyCorrelated = [
        "./03InverseStronglyCorrelated/n00050/R10000/s002.kp",
        "./03InverseStronglyCorrelated/n00100/R10000/s003.kp",
        "./03InverseStronglyCorrelated/n00200/R10000/s005.kp",
        "./03InverseStronglyCorrelated/n00500/R10000/s008.kp",
        "./03InverseStronglyCorrelated/n00500/R01000/s008.kp",
        "./03InverseStronglyCorrelated/n01000/R01000/s013.kp",
        "./03InverseStronglyCorrelated/n02000/R01000/s021.kp",
        "./03InverseStronglyCorrelated/n05000/R01000/s034.kp",
        "./03InverseStronglyCorrelated/n10000/R01000/s055.kp",
        "./03InverseStronglyCorrelated/n10000/R10000/s055.kp"
    ]
    path_04AlmostStronglyCorrelated = [
        "./04AlmostStronglyCorrelated/n00050/R10000/s002.kp",
        "./04AlmostStronglyCorrelated/n00100/R10000/s003.kp",
        "./04AlmostStronglyCorrelated/n00200/R10000/s005.kp",
        "./04AlmostStronglyCorrelated/n00500/R10000/s008.kp",
        "./04AlmostStronglyCorrelated/n00500/R01000/s008.kp",
        "./04AlmostStronglyCorrelated/n01000/R01000/s013.kp",
        "./04AlmostStronglyCorrelated/n02000/R01000/s021.kp",
        "./04AlmostStronglyCorrelated/n05000/R01000/s034.kp",
        "./04AlmostStronglyCorrelated/n10000/R01000/s055.kp",
        "./04AlmostStronglyCorrelated/n10000/R10000/s055.kp"
    ]
    path_05SubsetSum = [
        "./05SubsetSum/n00050/R10000/s002.kp",
        "./05SubsetSum/n00100/R10000/s003.kp",
        "./05SubsetSum/n00200/R10000/s005.kp",
        "./05SubsetSum/n00500/R10000/s008.kp",
        "./05SubsetSum/n00500/R01000/s008.kp",
        "./05SubsetSum/n01000/R01000/s013.kp",
        "./05SubsetSum/n02000/R01000/s021.kp",
        "./05SubsetSum/n05000/R01000/s034.kp",
        "./05SubsetSum/n10000/R01000/s055.kp",
        "./05SubsetSum/n10000/R10000/s055.kp"
    ]
    path_06UncorrelatedWithSimilarWeights = [
        "./06UncorrelatedWithSimilarWeights/n00050/R10000/s002.kp",
        "./06UncorrelatedWithSimilarWeights/n00100/R10000/s003.kp",
        "./06UncorrelatedWithSimilarWeights/n00200/R10000/s005.kp",
        "./06UncorrelatedWithSimilarWeights/n00500/R10000/s008.kp",
        "./06UncorrelatedWithSimilarWeights/n00500/R01000/s008.kp",
        "./06UncorrelatedWithSimilarWeights/n01000/R01000/s013.kp",
        "./06UncorrelatedWithSimilarWeights/n02000/R01000/s021.kp",
        "./06UncorrelatedWithSimilarWeights/n05000/R01000/s034.kp",
        "./06UncorrelatedWithSimilarWeights/n10000/R01000/s055.kp",
        "./06UncorrelatedWithSimilarWeights/n10000/R10000/s055.kp"
    ]
    path_07SpannerUncorrelated = [
        "./07SpannerUncorrelated/n00050/R10000/s002.kp",
        "./07SpannerUncorrelated/n00100/R10000/s003.kp",
        "./07SpannerUncorrelated/n00200/R10000/s010.kp",
        "./07SpannerUncorrelated/n00500/R10000/s008.kp",
        "./07SpannerUncorrelated/n00500/R01000/s008.kp",
        "./07SpannerUncorrelated/n01000/R01000/s013.kp",
        "./07SpannerUncorrelated/n02000/R01000/s021.kp",
        "./07SpannerUncorrelated/n05000/R01000/s034.kp",
        "./07SpannerUncorrelated/n10000/R01000/s055.kp",
        "./07SpannerUncorrelated/n10000/R10000/s055.kp"
    ]

    path_08SpannerWeaklyCorrelated = [
        "./08SpannerWeaklyCorrelated/n00050/R10000/s002.kp",
        "./08SpannerWeaklyCorrelated/n00100/R10000/s003.kp",
        "./08SpannerWeaklyCorrelated/n00200/R10000/s005.kp",
        "./08SpannerWeaklyCorrelated/n00500/R10000/s008.kp",
        "./08SpannerWeaklyCorrelated/n00500/R01000/s008.kp",
        "./08SpannerWeaklyCorrelated/n01000/R01000/s013.kp",
        "./08SpannerWeaklyCorrelated/n02000/R01000/s021.kp",
        "./08SpannerWeaklyCorrelated/n05000/R01000/s034.kp",
        "./08SpannerWeaklyCorrelated/n10000/R01000/s055.kp",
        "./08SpannerWeaklyCorrelated/n10000/R10000/s055.kp"
    ]
    path_09SpannerStronglyCorrelated = [
        "./09SpannerStronglyCorrelated/n00050/R10000/s002.kp",
        "./09SpannerStronglyCorrelated/n00100/R10000/s003.kp",
        "./09SpannerStronglyCorrelated/n00200/R10000/s005.kp",
        "./09SpannerStronglyCorrelated/n00500/R10000/s008.kp",
        "./09SpannerStronglyCorrelated/n00500/R01000/s008.kp",
        "./09SpannerStronglyCorrelated/n01000/R01000/s013.kp",
        "./09SpannerStronglyCorrelated/n02000/R01000/s021.kp",
        "./09SpannerStronglyCorrelated/n05000/R01000/s034.kp",
        "./09SpannerStronglyCorrelated/n10000/R01000/s055.kp",
        "./09SpannerStronglyCorrelated/n10000/R10000/s055.kp"
    ]
    path_10MultipleStronglyCorrelated = [
        "./10MultipleStronglyCorrelated/n00050/R10000/s002.kp",
        "./10MultipleStronglyCorrelated/n00100/R10000/s003.kp",
        "./10MultipleStronglyCorrelated/n00200/R10000/s005.kp",
        "./10MultipleStronglyCorrelated/n00500/R10000/s008.kp",
        "./10MultipleStronglyCorrelated/n00500/R01000/s008.kp",
        "./10MultipleStronglyCorrelated/n01000/R01000/s013.kp",
        "./10MultipleStronglyCorrelated/n02000/R01000/s021.kp",
        "./10MultipleStronglyCorrelated/n05000/R01000/s034.kp",
        "./10MultipleStronglyCorrelated/n10000/R01000/s055.kp",
        "./10MultipleStronglyCorrelated/n10000/R10000/s055.kp"
    ]
    path_11ProfitCeiling = [
        "./11ProfitCeiling/n00050/R10000/s002.kp",
        "./11ProfitCeiling/n00100/R10000/s003.kp",
        "./11ProfitCeiling/n00200/R10000/s005.kp",
        "./11ProfitCeiling/n00500/R10000/s008.kp",
        "./11ProfitCeiling/n00500/R01000/s008.kp",
        "./11ProfitCeiling/n01000/R01000/s013.kp",
        "./11ProfitCeiling/n02000/R01000/s021.kp",
        "./11ProfitCeiling/n05000/R01000/s034.kp",
        "./11ProfitCeiling/n10000/R01000/s055.kp",
        "./11ProfitCeiling/n10000/R10000/s055.kp"
    ]
    path_12Circle = [
        "./12Circle/n00050/R10000/s002.kp",
        "./12Circle/n00100/R10000/s003.kp",
        "./12Circle/n00200/R10000/s005.kp",
        "./12Circle/n00500/R10000/s008.kp",
        "./12Circle/n00500/R01000/s008.kp",
        "./12Circle/n01000/R01000/s013.kp",
        "./12Circle/n02000/R01000/s021.kp",
        "./12Circle/n05000/R01000/s034.kp",
        "./12Circle/n10000/R01000/s055.kp",
        "./12Circle/n10000/R10000/s055.kp"
    ]
    testcase = [
        path_00Uncorrelated,
        path_01WeaklyCorrelated,
        path_02StronglyCorrelated,
        path_03InverseStronglyCorrelated,
        path_04AlmostStronglyCorrelated,
        path_05SubsetSum,
        path_06UncorrelatedWithSimilarWeights,
        path_07SpannerUncorrelated,
        path_08SpannerWeaklyCorrelated,
        path_09SpannerStronglyCorrelated,
        path_10MultipleStronglyCorrelated,
        path_11ProfitCeiling,
        path_12Circle
    ]
    '''
    for test_folder in (testcase):
        for test_path in test_folder:
            print("\n============== Begin {}================\n".format(test_path))
            solver(test_path, 7*60)
            print("\n============== End ================\n\n")
    '''

    for path in testcase[12]:
        print("\n============== Begin {}================\n".format(path))
        solver(path, 7*60)
        print("\n============== End ================\n\n")


if __name__ == '__main__':
    main()
