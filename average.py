import sys

def main():
    
    total = 0.0
    length = 0.0
    average = 0.0

    try:
        #Get the name of a file
        filename = sys.argv[1]
        print(filename)
        #Open the file
        infile = open(filename, 'r')

        #Read values from file and compute average
        for line in infile:
            amount = float(line)
            total += amount
            length = length + 1

        average = total / length

        #Close the file
        infile.close()

        #Print the amount of numbers in file and average
        print('There were ', length, ' numbers in the file.' )
        print(format(average, ',.2f'))

    except IOError:
        print('An error occurred trying to read the file.')

    except ValueError:
        print('Non-numeric data found in the file')

    except:
        print('An error has occurred')


main()