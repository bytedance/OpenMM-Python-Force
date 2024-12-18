namespace OpenMM {
class Force {
public:
  virtual ~Force() ;
  int getForceGroup() const ;
  void setForceGroup(int group) ;
  const std::string& getName() const ;
  void setName(const std::string &name) ;
  virtual bool usesPeriodicBoundaryConditions() const ;
};
}
