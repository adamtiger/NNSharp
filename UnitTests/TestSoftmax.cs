using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestSoftmax
    {
        [TestMethod]
        public void Test_Softmax_Execute()
        {

            // Softmax output
            softmax = new SoftmaxLayer();
            Data2D data = new Data2D(1,1,5,1);

            data[0,0,0,0] = 0.0;
            data[0,0,1,0] = 1.0;
            data[0,0,2,0] = 1.5;
            data[0,0,3,0] = 2.0;
            data[0,0,4,0] = 3.0;

            softmax.SetInput(data);
            softmax.Execute();

            Data2D output = softmax.GetOutput() as Data2D;

            // Expected output
            double[] expOu = new double[5];

            double sum = 0.0;
            sum += (Math.Exp(0.0) + Math.Exp(1.0) + Math.Exp(1.5) + Math.Exp(2.0) + Math.Exp(3.0));

            expOu[0] = Math.Exp(0.0) / sum;
            expOu[1] = Math.Exp(1.0) / sum;
            expOu[2] = Math.Exp(1.5) / sum;
            expOu[3] = Math.Exp(2.0) / sum;
            expOu[4] = Math.Exp(3.0) / sum;

            Assert.AreEqual(output[0,0,0,0], expOu[0], 0.00000001);
            Assert.AreEqual(output[0,0,1,0], expOu[1], 0.00000001);
            Assert.AreEqual(output[0,0,2,0], expOu[2], 0.00000001);
            Assert.AreEqual(output[0,0,3,0], expOu[3], 0.00000001);
            Assert.AreEqual(output[0,0,4,0], expOu[4], 0.00000001);
        }

        private SoftmaxLayer softmax;

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullData()
        {
            DataArray data = null;
            SoftmaxLayer soft = new SoftmaxLayer();
            soft.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Softmax()
        {
            Data2D data = new Data2D(5,4,5,10);
            SoftmaxLayer soft = new SoftmaxLayer();
            soft.SetInput(data);
        }
    }
}
