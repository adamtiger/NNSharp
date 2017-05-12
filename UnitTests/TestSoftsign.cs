using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestSoftsign
    {
        [TestMethod]
        public void Test_Softsign_Execute()
        {
            relu = new SoftsignLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            relu.SetInput(data);

            relu.Execute();

            Data2D output = relu.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], SoftsignFunc(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], SoftsignFunc(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], SoftsignFunc(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], SoftsignFunc(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], SoftsignFunc(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], SoftsignFunc(-3.0), 0.00000001);
        }

        private double SoftsignFunc(double x)
        {
            return x / (1 + Math.Abs(x));
        }


        private SoftsignLayer relu;
    }
}
