import sys
sys.dont_write_bytecode = True

import sys
import matplotlib.pyplot as plt

model_name = sys.argv[1]

scores_file = open('./models/' + model_name + '/log/scores.txt', 'r')
errors_file = open('./models/' + model_name + '/log/std_errors.txt', 'r')

for line in scores_file:
	scores = list(map(float, line.strip().split(',')))
for line in errors_file:
	errors = list(map(float, line.strip().split(',')))

lowerCI = [scores[i] - 1*errors[i] for i in range(len(scores))]
upperCI = [scores[i] + 1*errors[i] for i in range(len(scores))]

# Get the confidence intervals of the model
# predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T

# Data for regions where we want to shade to indicate the intervals has
# to be sorted by the x axis to display correctly
# CI_df = pd.DataFrame(columns = ['x_data', 'low_CI', 'upper_CI'])
# CI_df['x_data'] = daily_data['temp']
# CI_df['low_CI'] = predict_mean_ci_low
# CI_df['upper_CI'] = predict_mean_ci_upp
# CI_df.sort_values('x_data', inplace = True)

# Define a function for the line plot with intervals
def lineplotCI(x_data, y_data, low_CI, upper_CI, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
    # Shade the confidence interval
    ax.fill_between(x_data, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '66% CI')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')	
    plt.axhline(y=900, linewidth=1, color='r')
    plt.savefig('./models/' + model_name + '/log/scores.png')
    plt.show()

# Call the function to create plot
lineplotCI(x_data = [10*i for i in range(len(scores))]
           , y_data = scores
           , low_CI = lowerCI
           , upper_CI = upperCI
           , x_label = 'Training episode'
           , y_label = 'Score over 10 random test tracks'
           , title = 'Performance without dropout')