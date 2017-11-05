using Newtonsoft.Json.Linq;
using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.IO;
using NNSharp.Models;

namespace UnitTests
{
    public class Utils
    {
        static public Data2D ReadDataFromFile(string filePath)
        {
            JObject model = JObject.Parse(File.ReadAllText(filePath));

            List<List<List<List<double>>>> weight = model.SelectToken("data").Select(
                            batch => batch.Select(
                                    row => row.Select(
                                            col => col.Select(
                                                    channel => (double)channel
                                                ).ToList()
                                        ).ToList()
                                ).ToList()
                        ).ToList();

            int batchNum = weight.Count;
            int rowNum = weight[0].Count;
            int colNum = weight[0][0].Count;
            int chNum = weight[0][0][0].Count;

            Data2D data = new Data2D(rowNum, colNum, chNum, batchNum);

            for (int row = 0; row < rowNum; ++row)
            {
                for (int col = 0; col < colNum; ++col)
                {
                    for (int chnl = 0; chnl < chNum; ++chnl)
                    {
                        for (int batch = 0; batch < batchNum; ++batch)
                        {
                            data[row, col, chnl, batch] = weight[batch][row][col][chnl];
                        }
                    }
                }
            }

            return data;
        }

        static public void CheckDimensions(Data2D actual, Data2D expected)
        {
            Dimension dimAct = actual.GetDimension();
            Dimension dimExp = expected.GetDimension();

            Assert.AreEqual(dimAct.b, dimExp.b);
            Assert.AreEqual(dimAct.h, dimExp.h);
            Assert.AreEqual(dimAct.w, dimExp.w);
            Assert.AreEqual(dimAct.c, dimExp.c);
        }

        static public void CheckResults(Data2D actual, Data2D expected, double accuracy)
        {
            Dimension dimExp = expected.GetDimension();

            for (int row = 0; row < dimExp.h; ++row)
            {
                for (int col = 0; col < dimExp.w; ++col)
                {
                    for (int chnl = 0; chnl < dimExp.c; ++chnl)
                    {
                        for (int batch = 0; batch < dimExp.b; ++batch)
                        {
                            Assert.AreEqual(actual[row, col, chnl, batch], expected[row, col, chnl, batch], accuracy);
                        }
                    }
                }
            }
        }

        static public void KerasModelTest(string pathIn, string pathModel, string pathOut, double accuracy = 0.00001)
        {
            var reader = new ReaderKerasModel(pathModel);
            SequentialModel model = reader.GetSequentialExecutor();

            // Initialize data.
            Data2D data = Utils.ReadDataFromFile(pathIn);

            // Load expected output and calculate the actual output.
            Data2D expected = Utils.ReadDataFromFile(pathOut);
            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Utils.CheckDimensions(output, expected);

            // Checking calculation
            Utils.CheckResults(output, expected, accuracy);
        }
    }
}
