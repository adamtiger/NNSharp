using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestSigmoid
    {
        [TestMethod]
        public void Test_Sigmoid_Execute()
        {
            sigmoid = new SigmoidLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            sigmoid.SetInput(data);

            sigmoid.Execute();

            Data2D output = sigmoid.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], SigmoidFunc(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], SigmoidFunc(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], SigmoidFunc(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], SigmoidFunc(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], SigmoidFunc(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], SigmoidFunc(-3.0), 0.00000001);
        }

        private double SigmoidFunc(double x)
        {
            return 1.0/(1 + Math.Exp(-x));
        }


        private SigmoidLayer sigmoid;
    }
}
