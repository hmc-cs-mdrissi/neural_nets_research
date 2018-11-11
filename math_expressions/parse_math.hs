
{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}

module LambdaParser where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim ((<|>), parse, getInput)
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Control.Applicative (some, many)
import Data.String
import Data.Function
import Data.Bifunctor
import Data.Maybe (fromMaybe)
import Text.Parsec.Error
import Debug.Trace (trace)
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import Data.Text (Text)
import Data.Map (Map)
import qualified Data.Map.Lazy as Map
import Data.ByteString.Lazy (writeFile)
import Prelude hiding (writeFile)
import GHC.Generics

--Let expressions are not part of the definition because let expressions can be viewed as syntactic sugar.
--let x = y in z is the same as (lambda x.z) y.
data MathExpression
  = IntegerM Integer
  | DoubleM Double
  | VarName Char
  | Nil
  | Symbol Symbol
  | Container ContainerSymbol MathExpression ContainerSymbol
  | UnOp UnOp MathExpression
  | PUnOp MathExpression PUnOp
  | DoubOp DoubOp MathExpression MathExpression
  | BinOp MathExpression BinOp MathExpression
  | Sum MathExpression MathExpression MathExpression
  | Integral MathExpression MathExpression MathExpression -- think about this more.  do we want bounds?  integraion vars?
  deriving (Show, Eq, Generic)

data ContainerSymbol = AbsBar | LeftParen | RightParen | LeftBrace | RightBrace| Magnitude deriving (Show, Eq, Generic)

data BinOp =
    Plus |
    Minus |
    Div |
    Mult |
    BinaryPm |
    Equal |
    Marrow |
    SubscriptOp |
    SuperscriptOp |
    ImplicitMult |
    Le |
    Leq |
    Ge |
    Geq |
    Neq deriving (Show, Eq, Generic)

data UnOp =
    Sin | Cos | Tan | Sqrt | NegSign | UnaryPm deriving (Show, Eq, Generic)

data DoubOp =
    FracOp | LogOp | LimOp deriving (Show, Eq, Generic)

data Symbol =
    Alpha |
    Beta |
    Gamma |
    Phi |
    Pi |
    Theta |
    Infty |
    Ldots deriving (Show, Eq, Generic)

data PUnOp = Factorial deriving (Show, Eq, Generic)

data BarOp = Bar deriving (Show, Eq, Generic) 


instance ToJSON MathExpression
instance ToJSON ContainerSymbol
instance ToJSON BinOp
instance ToJSON UnOp
instance ToJSON DoubOp
instance ToJSON Symbol
instance ToJSON PUnOp





mathDef :: P.LanguageDef ()
mathDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum <|> char '\''
                     ,P.opStart = oneOf "=.[];"
                     ,P.opLetter = oneOf "=/*"
                     ,P.reservedOpNames = ["}", "{", "$", "=", "+", "-", "d x", "\\int", "\\pm", "\\div","\\times", "\\sin", "\\cos", "\\tan", "!", "\\lim", "\\sum", "|", "\\log", "_", "\\frac", "\\alpha", "\\beta", "\\gamma", "\\phi", "\\pi", "\\theta", "\\infty", "\\ldots", "\\sqrt"]
                     ,P.reservedNames = []
                     }

integralP :: Parser ()
integralP = P.reservedOp lexer "\\int"

dxP :: Parser ()
dxP = P.reservedOp lexer "d x"

alphaP :: Parser ()
alphaP = P.reservedOp lexer "\\alpha"

betaP :: Parser ()
betaP = P.reservedOp lexer "\\beta"

gammaP :: Parser ()
gammaP = P.reservedOp lexer "\\gamma"

phiP :: Parser ()
phiP = P.reservedOp lexer "\\phi"

piP :: Parser ()
piP = P.reservedOp lexer "\\pi"

thetaP :: Parser ()
thetaP = P.reservedOp lexer "\\theta"

inftyP :: Parser ()
inftyP = P.reservedOp lexer "\\infty"

ldotsP :: Parser ()
ldotsP = P.reservedOp lexer "\\ldots"


lexer :: P.TokenParser ()
lexer = P.makeTokenParser mathDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

equal :: Parser ()
equal = P.reservedOp lexer "="

factorial :: Parser ()
factorial = P.reservedOp lexer "!"

plus :: Parser ()
plus = char '+' *> return ()

minus :: Parser ()
minus = P.reservedOp lexer "-"

divide :: Parser ()
divide = P.reservedOp lexer "\\div"

mult :: Parser ()
mult = P.reservedOp lexer "\\times"

marrow :: Parser ()
marrow = P.reservedOp lexer "\\rightarrow"

identifier :: Parser Char
identifier = letter

lexeme :: Parser a -> Parser a
lexeme = P.lexeme lexer

sinP :: Parser ()
sinP = P.reservedOp lexer "\\sin"

cosP :: Parser ()
cosP = P.reservedOp lexer "\\cos"

tanP :: Parser ()
tanP = P.reservedOp lexer "\\tan"

sqrtP :: Parser ()
sqrtP = P.reservedOp lexer "\\sqrt"

geqP :: Parser ()
geqP = P.reservedOp lexer "\\geq"

geP :: Parser ()
geP = P.reservedOp lexer ">"

neqP :: Parser ()
neqP = P.reservedOp lexer "\\neq"

sumP :: Parser ()
sumP = P.reservedOp lexer "\\sum"

leqP :: Parser ()
leqP = P.reservedOp lexer "\\leq"

leP :: Parser ()
leP = P.reservedOp lexer "<"

equalP :: Parser ()
equalP = P.reservedOp lexer "="

absP :: Parser ()
absP = P.reservedOp lexer "|"

magP :: Parser ()
magP = P.reservedOp lexer "||"

lParenP :: Parser ()
lParenP = P.reservedOp lexer "\\left(" 

rParenP :: Parser ()
rParenP = P.reservedOp lexer "\\right)"

lBraceP :: Parser ()
lBraceP = P.reservedOp lexer "{"

rBraceP :: Parser ()
rBraceP = char '}' *> return ()

logP :: Parser()
logP = P.reservedOp lexer "\\log"

limP :: Parser()
limP = P.reservedOp lexer "\\lim"

fracP :: Parser()
fracP = P.reservedOp lexer "\\frac"

underscoreP :: Parser()
underscoreP = P.reservedOp lexer "_"

dollar :: Parser()
dollar = P.reservedOp lexer "$"

pm :: Parser()
pm = P.reservedOp lexer "\\pm"

superscriptP :: Parser()
superscriptP = P.reservedOp lexer "^"

symbol :: Parser MathExpression
symbol = (do sym <- alphaP
             return $ Symbol Alpha)
          <|> 
          (do sym <- betaP
              return $ Symbol Beta)
          <|> 
          (do sym <- gammaP
              return $ Symbol Gamma)
          <|> 
          (do sym <- phiP
              return $ Symbol Phi)
          <|> 
          (do sym <- piP
              return $ Symbol Pi)
          <|> 
          (do sym <- inftyP
              return $ Symbol Infty)
          <|> 
          (do sym <- ldotsP
              return $ Symbol Ldots)


varname :: Parser MathExpression
varname = do name <- identifier
             spaces
             return $ VarName name

whiteSpace :: Parser ()
whiteSpace = P.whiteSpace lexer

number :: Parser MathExpression
number = do num <- P.naturalOrFloat lexer
            case num of
                Left x -> return $ IntegerM x
                Right x -> return $ DoubleM x


mathExpression' :: Int -> Bool -> Bool -> Bool -> Parser MathExpression
mathExpression' 0 bool sumBool absBool = (mathExpression' 1 bool sumBool absBool `chainl1` (pure (\x -> BinOp x ImplicitMult)))
mathExpression' 1 bool sumBool absBool = mathExpression' 2 bool sumBool absBool `chainl1` (
                        (leP *> pure (\x -> BinOp x Le)) <|> 
                        (leqP *> pure (\x -> BinOp x Leq)) <|> 
                        (geP *> pure (\x -> BinOp x Ge)) <|> 
                        (geqP *> pure (\x -> BinOp x Geq)) <|> 
                        (equalP *> pure (\x -> BinOp x Equal)) <|> 
                        (neqP *> pure (\x -> BinOp x Neq) )) -- all the comparison ops  :)
mathExpression' 2 bool sumBool absBool = mathExpression' 3 bool sumBool absBool `chainl1` ((marrow *> pure (\x -> BinOp x Marrow))) -- arrow :)
mathExpression' 3 bool sumBool absBool = mathExpression' 4 bool sumBool absBool `chainl1` ((plus *> pure (\x -> BinOp x Plus)) <|> (minus *> pure (\x -> BinOp x Minus)) <|> (pm *> pure (\x -> BinOp x BinaryPm))) -- plus, minus  :)
mathExpression' 4 bool sumBool absBool = mathExpression' 5 bool sumBool absBool `chainl1` ((mult *> pure (\x -> BinOp x Mult)) <|> (divide *> pure (\x -> BinOp x Div))) -- times, div :)
mathExpression' 5 bool sumBool absBool = do exp <- mathExpression' 6 bool sumBool absBool
                                            facs <- many factorial
                                            return $ iterate (flip PUnOp Factorial) exp !! (length facs) -- factorial
mathExpression' 6 bool sumBool absBool = do minusSigns <- many minus
                                            exp <- mathExpression' 65 bool sumBool absBool ----- 777
                                            return $ iterate (UnOp NegSign) exp !! (length minusSigns) -- negate
mathExpression' 65 bool sumBool absBool = (do pmSign <- many pm
                                              exp <- mathExpression' 8 bool sumBool absBool ----- 777
                                              return (UnOp NegSign exp))
                                          <|> 
                                          mathExpression' 8 bool sumBool absBool
                       

mathExpression' 7 bool sumBool absBool= (do sinS <- some sinP 
                                            exp <- mathExpression' 9 bool sumBool absBool ---888
                                            return $ iterate (UnOp Sin) exp !! (length sinS))
                <|> 
                    (do cosS <- some cosP 
                        exp <- mathExpression' 7 bool sumBool absBool
                        return $ iterate (UnOp Cos) exp !! (length cosS))
                <|> 
                    (do tanS <- some tanP 
                        exp <- mathExpression' 7 bool sumBool absBool
                        return $ iterate (UnOp Tan) exp !! (length tanS))
                <|> 
                    (do sqrtS <- some sqrtP 
                        exp <- mathExpression' 7 bool sumBool absBool
                        return $ iterate (UnOp Sqrt) exp !! (length sqrtS))
                <|>
                    (do logP
                        ((do res <- underscoreP
                             logB <- mathExpression' 7  False sumBool absBool --TODO: CHANGE THESE BACK!!!
                             logV <- mathExpression' 7 False sumBool absBool 
                             return $ DoubOp LogOp logB logV) <|>
                         (do logV <- mathExpression' 7 False sumBool absBool 
                             return $ DoubOp LogOp Nil logV)))
                <|> 
                   (do limP
                       underscoreP
                       limB <- mathExpression' 7 False sumBool absBool
                       limV <- mathExpression' 7 True sumBool absBool
                       return $ DoubOp LimOp limB limV)
                <|> 
                   (do fracP
                       numerator <- mathExpression' 7 False sumBool absBool
                       denominator <- mathExpression' 7 False sumBool absBool
                       return $ DoubOp FracOp numerator denominator)
                <|> 
                   (do sumP
                       lowerBound <- optionMaybe (do underscoreP
                                                     mathExpression' 7 bool True absBool)
                       upperBound <- optionMaybe (do superscriptP
                                                     mathExpression' 7 bool True absBool)
                       bodyValue <- mathExpression' 0 bool sumBool absBool
                       let lowerBoundValue = fromMaybe Nil lowerBound
                       let upperBoundValue = fromMaybe Nil upperBound
                       return $ Sum lowerBoundValue upperBoundValue bodyValue)
                <|>
                    (do integralP
                        body <- mathExpression' 7 False sumBool absBool 
                        dxP 
                        return $ Integral Nil Nil body)
                <|>
                  mathExpression' 9 bool sumBool absBool ---888

mathExpression' 8 bool sumBool absBool =  if sumBool then  --- 999
                                            (mathExpression' 7 False sumBool absBool `chainl1` (underscoreP *> pure (\x -> BinOp x SubscriptOp))) else 
                                            (mathExpression' 7 bool sumBool absBool `chainl1` (((underscoreP *> pure (\x -> BinOp x SubscriptOp))) <|> 
                                                                                                (trace (show bool) superscriptP *> pure (\x -> BinOp x SuperscriptOp))))

-- subscript, superscript  :)             
mathExpression' 9 bool sumBool absBool = (if (bool && not sumBool) then ((varname <|> number <|> symbol) `chainl1` (pure (\x -> BinOp x ImplicitMult))) else (varname <|> number <|> symbol))
                    <|> 
                       case absBool of 
                         True -> fail "nothing is working"
                         False -> (do
                                      magP
                                      expr <- mathExpression' 0 True False True
                                      magP
                                      return $ Container Magnitude expr Magnitude)
                    <|> 
                       case absBool of 
                         True -> fail "oh no"
                         False -> (do
                                      absP
                                      expr <- mathExpression' 0 True False True
                                      trace "final absp" absP
                                      return $ Container AbsBar expr AbsBar)
                   <|>
                       (do lParenP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rParenP
                           return $ Container LeftParen expr RightParen)
                    <|> 
                       (do lBraceP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rBraceP
                           return $ Container LeftBrace expr RightBrace)

parseMath :: String -> Either ParseError MathExpression
parseMath = parse (optional dollar *> mathExpression' 0 True False False) ""
-- parseMath = parse (whiteSpace *> between (char '$') (char '$') (mathExpression' 0 True False False) <* whiteSpace) ""
-- parseMath str = parse (whiteSpace *> mathExpression' 0 True False False) "" (tail str)

parseString =  show (parseMath "--\\frac{-3}{1}")


-- parseMath :: String -> MathExpression
-- parseMath str = Nil





-- data MathEntry = MathEntry {
--   fileName :: String,
--   latexExpr :: String,
-- }





-- THEN DO NEGATIVE SIGN

-- === bigger dataset ===
-- INTEGRALS!!!
-- brackets
-- mbox




-- main :: IO()
-- main = do 
--   a <- B.readFile "single_expr.json"
--   -- a <- B.readFile "TEST2016_INKML_GT_GET_Strings.json"
--   case decode a :: Maybe (Map String String) of
--     Just loadedExprs -> writeFile "just_testing.json" $ encode (toJSON (Map.map (parseMath) (loadedExprs)))
--     Nothing -> print "Unparsable"
--   putStrLn "hi"

-- main = putStrLn parseString
-- loadedExprs is a map from the String to (Either ParseError MathExpression)
-- sequence is... something???  WE REALLY WANT IT TO GO FROM (WHATEVER) TO (TOJSON)
-- encode :: ToJSON a => a -> ByteString
-- writeFile:: FilePath -> ByteString -> IO ()




