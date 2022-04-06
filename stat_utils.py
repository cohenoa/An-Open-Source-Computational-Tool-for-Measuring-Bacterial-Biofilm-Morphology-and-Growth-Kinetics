import scipy


def print_num_repetitions(df):
    for d in [1.0, 1.5, 2.0, 3.0]:
        print('Distance From CHX {} cm: {} plates'.format(d, len(df[(df['DistanceFromCHX'] == d) & (df['Day'] == 3)])))


def print_stats_table(df, column_of_interest, days):

    for day in days:
        values_control = df[(df['DistanceFromCHX'] == 3.0) & (df['Day'] == day)][column_of_interest]
        values_1cm = df[(df['DistanceFromCHX'] == 1.0) & (df['Day'] == day)][column_of_interest]
        values_15cm = df[(df['DistanceFromCHX'] == 1.5) & (df['Day'] == day)][column_of_interest]
        values_2cm = df[(df['DistanceFromCHX'] == 2.0) & (df['Day'] == day)][column_of_interest]

        # 1 cm to control:
        print('[{}]: Day {}, 1 vs. control: {:.5f}'.format(column_of_interest, day,
                                                           scipy.stats.ttest_ind(values_1cm, values_control).pvalue))
        # 1.5 cm to control:
        print('[{}]: Day {}, 1.5 vs. control: {:.5f}'.format(column_of_interest, day,
                                                           scipy.stats.ttest_ind(values_15cm, values_control).pvalue))

        # 2 cm to control:
        print('[{}]: Day {}, 2 vs. control: {:.5f}'.format(column_of_interest, day,
                                                           scipy.stats.ttest_ind(values_2cm, values_control).pvalue))
        print('----------------------------------')


def figure_4_compute_pvalues(df_day3):
    # CORE:
    # 1 cm to control:
    pvalue_1cm_to_control_core = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 1) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'])

    # 1.5 cm to control:
    pvalue_1_5cm_to_control_core = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 1.5) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'])

    # 2 cm to control:
    pvalue_2cm_to_control_core = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 2) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Core')]['ratio_right_to_left_intensity'])

    # PERIPHERY:
    pvalue_1cm_to_control_periphery = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 1) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'])

    pvalue_1_5cm_to_control_periphery = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 1.5) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'])

    pvalue_2cm_to_control_periphery = scipy.stats.ttest_ind(
        df_day3[(df_day3['DistanceFromCHX'] == 2) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'], \
        df_day3[(df_day3['DistanceFromCHX'] == 3) & (df_day3['Region type'] == 'Periphery')]['ratio_right_to_left_intensity'])


    print('1 cm core: {:.2f}, periphery: {:.2f} '.format(pvalue_1cm_to_control_core.pvalue,
                                                         pvalue_1cm_to_control_periphery.pvalue))
    print('1.5 cm core: {:.2f}, periphery: {:.2f} '.format(pvalue_1_5cm_to_control_core.pvalue,
                                                           pvalue_1_5cm_to_control_periphery.pvalue))
    print('2 cm core: {:.2f}, periphery: {:.2f} '.format(pvalue_2cm_to_control_core.pvalue,
                                                         pvalue_2cm_to_control_periphery.pvalue))



