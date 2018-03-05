generateArbitrary: GenerateArbitraryTests.hs ArbitraryTests.hs LambdaParser.hs ForParser.hs ForLambdaCommon.hs
	ghc GenerateArbitraryTests.hs -main-is GenerateArbitraryTests.main -o generateArbitrary

clean:
	rm -f *.o *.hi generateArbitrary

cleanjson:
	rm -f *.json