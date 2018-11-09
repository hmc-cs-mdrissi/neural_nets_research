module Parser where

import Types

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

mathDef :: P.LanguageDef ()
mathDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum <|> char '\''
                     ,P.opStart = oneOf "=.[];"
                     ,P.opLetter = oneOf "=+/*"
                     ,P.reservedOpNames = ["=", "+", "-", "/","*", "\\sin", "\\cos", "\\tan", "!", "\\lim", "\\sum", "|", "\\log", "_", "\\frac", "\\alpha", "\\beta", "\\gamma", "\\phi", "\\pi", "\\theta", "\\infty", "\\ldots", "\\sqrt"]
                     ,P.reservedNames = []
                     }

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
plus = P.reservedOp lexer "+"

minus :: Parser ()
minus = P.reservedOp lexer "-"

divide :: Parser ()
divide = P.reservedOp lexer "/"

mult :: Parser ()
mult = P.reservedOp lexer "*"

-- negation :: Parser ()
-- negation = P.reservedOp lexer "~"

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
lParenP = P.reservedOp lexer "(" 

rParenP :: Parser ()
rParenP = P.reservedOp lexer ")"

lBraceP :: Parser ()
lBraceP = P.reservedOp lexer "{"

rBraceP :: Parser ()
rBraceP = P.reservedOp lexer "}"

logP :: Parser()
logP = P.reservedOp lexer "\\log"

limP :: Parser()
limP = P.reservedOp lexer "\\lim"

fracP :: Parser()
fracP = P.reservedOp lexer "\\frac"







underscoreP :: Parser()
underscoreP = P.reservedOp lexer "_"

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
mathExpression' 0 bool sumBool absBool = (mathExpression' 1 bool sumBool absBool `chainl1` (pure (\x -> NatBinOp x ImplicitMult)))
mathExpression' 1 bool sumBool absBool = mathExpression' 2 bool sumBool absBool `chainl1` (
                        (leP *> pure (\x -> NatBinOp x Le)) <|> 
                        (leqP *> pure (\x -> NatBinOp x Leq)) <|> 
                        (geP *> pure (\x -> NatBinOp x Ge)) <|> 
                        (geqP *> pure (\x -> NatBinOp x Geq)) <|> 
                        (equalP *> pure (\x -> NatBinOp x Equal)) <|> 
                        (neqP *> pure (\x -> NatBinOp x Neq) )) -- all the comparison ops  :)
mathExpression' 2 bool sumBool absBool = mathExpression' 3 bool sumBool absBool `chainl1` ((marrow *> pure (\x -> NatBinOp x Marrow))) -- arrow :)
mathExpression' 3 bool sumBool absBool = mathExpression' 4 bool sumBool absBool `chainl1` ((trace "plus" $ plus *> pure (\x -> NatBinOp x Plus)) <|> (trace "minus" $ minus *> pure (\x -> NatBinOp x Minus))) -- plus, minus  :)
mathExpression' 4 bool sumBool absBool = mathExpression' 5 bool sumBool absBool `chainl1` ((trace "mul" $ mult *> pure (\x -> NatBinOp x Mult)) <|> (trace "div" $ divide *> pure (\x -> NatBinOp x Div))) -- times, div :)
mathExpression' 5 bool sumBool absBool = do exp <- mathExpression' 6 bool sumBool absBool
                                            facs <- many factorial
                                            return $ iterate (flip PUnOp Factorial) exp !! (length facs) -- factorial
mathExpression' 6 bool sumBool absBool = trace "negate" $ do minusSigns <- many minus
                                                             exp <- mathExpression' 7 bool sumBool absBool
                                                             return $ iterate (UnOp NegSign) exp !! (length minusSigns) -- negate
                       

mathExpression' 7 bool sumBool absBool= (do sinS <- many sinP 
                                            exp <- mathExpression' 8 bool sumBool absBool
                                            return $ iterate (UnOp Sin) exp !! (length sinS))
                <|> 
                    (do cosS <- many cosP 
                        exp <- mathExpression' 8 bool sumBool absBool
                        return $ iterate (UnOp Cos) exp !! (length cosS))
                <|> 
                    (do tanS <- many tanP 
                        exp <- mathExpression' 8 bool sumBool absBool
                        return $ iterate (UnOp Tan) exp !! (length tanS))
                <|> 
                    (do sqrtS <- many sqrtP 
                        exp <- mathExpression' 8 bool sumBool absBool
                        return $ iterate (UnOp Sqrt) exp !! (length sqrtS))
mathExpression' 8 bool sumBool absBool = if sumBool then (mathExpression' 9 False sumBool absBool `chainl1` (underscoreP *> pure (\x -> NatBinOp x SubscriptOp))) else (mathExpression' 9 False sumBool absBool `chainl1` (((underscoreP *> pure (\x -> NatBinOp x SubscriptOp))) <|> (superscriptP *> pure (\x -> NatBinOp x SuperscriptOp))))

-- subscript, superscript  :)             
mathExpression' 9 bool sumBool absBool = (if bool then ((varname <|> number <|> symbol) `chainl1` (pure (\x -> NatBinOp x ImplicitMult))) else (varname <|> number))
                    <|>
                    (do logP
                        ((do res <- underscoreP
                             logB <- mathExpression' 0 False False False
                             logV <- mathExpression' 0 False False False
                             return $ DoubOp LogOp logB logV) <|>
                         (do logV <- mathExpression' 0 False False False
                             return $ DoubOp LogOp Nil logV)))
                    <|> 
                       (do trace "sum" sumP
                           lowerBound <- optionMaybe (do underscoreP
                                                         mathExpression' 0 False True False)
                           upperBound <- optionMaybe (do superscriptP
                                                         mathExpression' 0 False False False)
                           bodyValue <- mathExpression' 0 True False False
                           let lowerBoundValue = fromMaybe Nil lowerBound
                           let upperBoundValue = fromMaybe Nil upperBound
                           return $ Sum lowerBoundValue upperBoundValue bodyValue)  
                    <|> 
                       (do trace "lim" limP
                           underscoreP
                           limB <- mathExpression' 0 False False False
                           limV <- mathExpression' 0 True False False
                           return $ DoubOp LimOp limB limV)
                    <|> 
                       (do trace "frac" fracP
                           numerator <- mathExpression' 0 False False False
                           denominator <- mathExpression' 0 False False False
                           return $ DoubOp FracOp numerator denominator)
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
                                      absP
                                      return $ Container AbsBar expr AbsBar)
                   <|>
                       (do trace "paren" lParenP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rParenP
                           return $ Container LeftParen expr RightParen)
                    <|> 
                       (do trace "last" lBraceP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rBraceP
                           return $ Container LeftBrace expr RightBrace)

parseMath :: String -> Either ParseError MathExpression
parseMath str = parse (whiteSpace *> mathExpression' 0 True False False) "" (tail (tail str))

parseString =  show (parseMath "--\\sqrt{-3}")

-- data MathEntry = MathEntry {
--   fileName :: String,
--   latexExpr :: String,
-- }





-- THEN DO NEGATIVE SIGN

-- === bigger dataset ===
-- INTEGRALS!!!
-- brackets
-- mbox

