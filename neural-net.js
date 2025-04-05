const dfd = require("danfojs-node");
var zeros = require( '@stdlib/array/base/zeros' );
var zeros2d = require( '@stdlib/array/base/zeros2d' );
var min = require( '@stdlib/math/base/special/min' );
var max = require( '@stdlib/math/base/special/max' );
var array2iterator = require( '@stdlib/array/to-iterator' );
var iterSubtract = require( '@stdlib/math/iter/ops/subtract' );
var uniform = require( '@stdlib/random/base/uniform' );
var pow = require( '@stdlib/math/base/special/pow' );
var ddot = require( '@stdlib/blas/base/ddot' );
var Float64Array = require( '@stdlib/array/float64' );
var ln = require( '@stdlib/math/base/special/ln' );
var abs = require( '@stdlib/math/base/special/abs' );
var sqrt = require( '@stdlib/math/base/special/sqrt' );



// Function to count occurance of numbers in an array
function countOccurrences(array) {
    const countMap = new Map();
    array.forEach(element => {
        if (countMap.has(element)) {
        countMap.set(element, countMap.get(element) + 1); 
        } else {
        countMap.set(element, 1); 
        }
    });
    countMap.forEach((value, key) => {
        console.log(`${key} -> ${value}`);
    });
}

// Function to randomize arrangement of dataset
function shuffleRows(matrix) {
    for (var i = matrix.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [matrix[i], matrix[j]] = [matrix[j], matrix[i]]; 
    }
    return matrix;
}


// Initialising Random weights and biases for neural network
var w1 = zeros2d([17,11]);
for(var i=0; i<w1.length; i++) for(var j=0; j<w1[0].length; j++) w1[i][j]=uniform( 0.0, 1.0 );

var b1 = zeros(17);
for(var i=0; i<b1.length; i++) b1[i]=uniform(0.0,1.0);

var wo = zeros2d([10,17]);
for(var i=0; i<wo.length; i++) for(var j=0; j<wo[0].length; j++) wo[i][j]=uniform( 0.0, 1.0 );

var bo = zeros(10);
for(var i=0; i<bo.length; i++) bo[i]=uniform(0.0,1.0);


// ReLU activation function
function ReLU(x) {
    for(var i=0; i<x.length; i++) {
        x[i]=max(0,x[i]);
    }

    return x;
}

// Softmax activation function 
function softmax(x) {
    var sum1 = 0.0;
    for(var j=0; j<x.length; j++) {
        sum1 += abs(x[j]);
    }
    for(var j=0; j<x.length; j++) {
        x[j] = abs(x[j])/sum1;
    }
    return x;
}


// Forward propagation through a layer function
function forward(w, b, input) {
    var x = new Float64Array(input);
    // console.log(input)
    for(var i=0; i<b.length; i++) {
        var y = new Float64Array(w[i]);
        var z = ddot( x.length, x, 1, y, 1 );
        b[i] += z;
    }
    return b;
}


// Loss function (cross-entropy)
function loss(output, actual) {
    const epsilon = 1e-10;
    var ls = 0;
    for(var i=0; i<output.length; i++) {
        var clipped_output = Math.max(epsilon, min(1-epsilon, output[i]));
        ls += abs( (actual[i]*ln(clipped_output) + (1-actual[i])*ln(1-clipped_output)));
    }
    ls /= output.length;
    return ls;
}


// Neural network function
function neu_net(x_train, y_train, alpha){

    // First Layer forward prop
    var l1 = forward(w1,b1,x_train);
    var l1_a = ReLU(l1);
    
    // Second layer forward prop
    var op = forward(wo, bo, l1_a);
    var op_s = softmax(op);

    y_train_a = zeros(10);
    y_train_a[y_train-1] = 1
    y_train = y_train_a;

    // Calculating loss
    ls = loss(op_s, y_train)


    // Backpropogation
    var dz2 = zeros(y_train.length);
    for(var i=0; i<y_train.length; i++) {
        dz2[i] = y_train[i]-op_s[i];

    }

    var dw2 = zeros2d([10,17]) 
    for(var i=0; i<17; i++) {
        for(var j=0; j<10; j++) {
            dw2[j][i]= dz2[j]*l1_a[i];
        }
    }

    var db2 = dz2;

    var dz1 = zeros(17); 
    for(var i=0; i<17; i++) {
        var temp = 0;
        for(var j=0; j<10; j++) {
            temp += wo[j][i]*dz2[j];
        }
        dz1[i] = temp;

        if(l1[i]<0) dz1[i] = 0.0; 
    }

    var dw1 = zeros2d([17,11]);
    for(var i=0; i<17; i++) {
        for(var j=0; j<11; j++) {
            dw1[i][j] = dz1[i]*x_train[j];
        }
    }

    var db1 = dz1;

    // updating all weights and biases together
    for(var i=0; i<wo.length; i++) {
        for(var j=0; j<wo[0].length; j++) {
            wo[i][j] -= alpha*dw2[i][j];
        }
    }

    for(var i=0; i<bo.length; i++) {
        bo[i] -= alpha*db2[i];
    }

    for(var i=0; i<w1.length; i++) {
        for(var j=0; j<w1[0].length; j++) {
            w1[i][j] -= alpha*dw1[i][j];


        }
    }

    for(var i=0; i<b1.length; i++) {
        b1[i] -= alpha*db1[i];

    }

    return ls
}

// Error calculating function for training batch 
function test_error(x_test, y_test) {
    output = ReLU(forward(w1,b1,x_test));
    output = softmax(forward(wo,bo,output));
    
    var maxi = 0;
    var index = -1;
    for(var i=0; i<output.length; i++) {
        if(maxi<output[i]) {
            maxi = output[i];
            index = i+1;
        }
    }

    index = abs(y_test-index);
    return index;
}


dfd.readCSV('winequality-red.csv')
    .then(df => {
        
        console.log("Columns in the dataset =>\n",df.columns)

        var data = [ ];
        for(var i=0; i<df.shape[0]; i++) {
            data.push(df.iloc({rows: [i], columns: [":"]}).values[0])
            
        }
    

        data = shuffleRows(data);

        console.log("\nOccurances of the target values (quality) => ")
        countOccurrences(df.quality.values);

        for(var i=0; i<df.shape[0]; i++) {
            data[i].pop();
        }

        var x = data;
        var y = df.quality.values;

        var arr_min = x[0];
        var arr_max = zeros(x[0].length);

        for(var i=0; i<x.length; i++) {
            for(var j=0; j<x[0].length; j++) {
                arr_min[j] = min(arr_min[j], x[i][j]);
                arr_max[j] = max(arr_max[j], x[i][j]);

            }
        }

        // Min-max scaling implementation
        for(var i=0; i<x.length; i++){
            var num = [];
            var den = [];
            var scal = [];
            var it1 = array2iterator(x[i]);
            var it2 = array2iterator(arr_min); 
            var it5 = iterSubtract(it1,it2);
            for(var j=0; j<x[0].length; j++) {
                num.push(it5.next().value)
            }


            it2 = array2iterator(arr_min); 
            var it3 = array2iterator(arr_max); 
            var it4 = iterSubtract( it3, it2 );
            for(var j=0; j<x[0].length; j++) {
                den.push(it4.next().value)
            }
            
            for(var j=0; j<x[0].length; j++) {
                scal.push(num[j]/den[j])
            }
            x[i] = scal;

        }

        var x_train = []
        var x_test = []
        var y_train = []
        var y_test = []


        for(var i=0; i<x.length*0.8; i++) {
            x_train.push(x[i])
            y_train.push(y[i]) 
        }   

        for(var i=x.length*0.8 + 1; i<x.length; i++) {
            x_test.push(x[Math.floor(i)])
            y_test.push(y[Math.floor(i)]) 
        }
 
        var alpha = 0.000001;
        var num_epoch = 20;
        console.log("\nTraining neural network ->");

        for(var i=0; i<num_epoch; i++) {
            var loss = 0;
            for(var j=0; j<x_train.length; j++) {
                loss += neu_net(x_train[j], y_train[j], alpha);
            }
            console.log("Loss in Epoch ", i+1, " : ", loss/x_train.length);
        }
        

        var mse  = 0.0;
        for(var i=0; i<y_test.length; i++) {
            mse += pow(test_error(x_test[i],y_test[i]),2);
        }
        var rmse = sqrt(mse/x_test.length);

        console.log("Root mean square error : ", rmse)

    })
    .catch(err => {
        console.log(err)
    })
